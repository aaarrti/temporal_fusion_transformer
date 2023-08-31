from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Generator, Literal, Tuple

import jax
from absl import logging
from absl_extra import flax_utils
from absl_extra.typing_utils import ParamSpec
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp

from temporal_fusion_transformer.src.config_dict import ConfigDictProto, DatasetConfig
from temporal_fusion_transformer.src.metrics import MetricContainer
from temporal_fusion_transformer.src.quantile_loss import make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.training_lib import (
    TrainStateContainer,
    load_dataset,
    make_optimizer,
    make_training_hooks,
    multi_device_train_step,
    multi_device_validation_step,
    single_device_train_step,
    single_device_validation_step,
)

P = ParamSpec("P")

if TYPE_CHECKING:
    import tensorflow as tf


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def train_experiment(
    *,
    data_dir: str,
    batch_size: int,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDictProto,
    data_config: DatasetConfig,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    device_type: Literal["cpu", "gpu", "tpu"] = "gpu",
    prefetch_buffer_size: int = 0,
    dynamic_scale: DynamicScale | None = None,
    verbose: bool = True,
    hooks: flax_utils.TrainingHooks | Callable[[int], flax_utils.TrainingHooks] | None = None,
    profile: bool = False,
) -> flax_utils.MetricsAndParams:
    if mixed_precision:
        if device_type == "gpu":
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.bfloat16
    else:
        compute_dtype = jnp.float32

    num_devices = jax.device_count()

    data = load_dataset(
        f"{data_dir}/{experiment_name}",
        batch_size * num_devices,
        config.prng_seed,
        dtype=compute_dtype,
        shuffle_buffer_size=config.shuffle_buffer_size,
        num_encoder_steps=data_config.num_encoder_steps,
    )
    tensorboard_logdir = f"tensorboard/{experiment_name}"
    return train(
        data=data,
        data_config=data_config,
        device_type=device_type,
        save_path=save_path,
        jit_module=jit_module,
        mixed_precision=mixed_precision,
        dynamic_scale=dynamic_scale,
        prefetch_buffer_size=prefetch_buffer_size,
        epochs=epochs,
        verbose=verbose,
        config=config,
        batch_size=batch_size,
        hooks=hooks,
        profile=profile,
        tensorboard_logdir=tensorboard_logdir,
    )


def train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDictProto,
    data_config: DatasetConfig,
    epochs: int = 1,
    batch_size: int | None = None,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    device_type: Literal["gpu", "tpu"] = "gpu",
    dynamic_scale: DynamicScale | None | Literal["auto"] = None,
    prefetch_buffer_size: int = 0,
    hooks: flax_utils.TrainingHooks | Callable[[int], flax_utils.TrainingHooks] | None = None,
    verbose: bool = True,
    tensorboard_logdir: str = None,
    profile: bool = False,
    early_stopping: EarlyStopping | None | Literal["auto"] = "auto",
) -> flax_utils.MetricsAndParams:
    """

    Parameters
    ----------
    data:
        Tuple of training & validation tf.data.Dataset's
    config:
    data_config:

    batch_size:
        Batch size for each device.
    epochs:
        Number of epochs to train.
    mixed_precision:
        Enable mixed float16 computations yes/no.
    jit_module:
        Apply nn.jit to model yes/no.
    save_path:
        Filename to save weights, if set to None weights will not be saved.
    verbose:
        Show progressbar yes/no.
    prefetch_buffer_size:
        Number of element to prefetch to the GPU (mb we don't need it at all)
    hooks:
        Custom training hooks.
    dynamic_scale:
        Custom loss scale to use for mixed float16 training.
    device_type:
        Relevant only for multi-device setting.
    tensorboard_logdir:
    profile
    early_stopping


    Returns
    -------

    """

    if mixed_precision:
        if device_type == "gpu":
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.bfloat16
    else:
        compute_dtype = jnp.float32

    multi_device = jax.device_count() > 1
    training_dataset, validation_dataset = data

    num_training_steps = int(training_dataset.cardinality())
    first_x = training_dataset.as_numpy_iterator().next()[0]

    if batch_size is not None:
        first_x = jnp.asarray(first_x[:batch_size], dtype=compute_dtype)

    model = TemporalFusionTransformer.from_config_dict(config, data_config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    params = model.init(params_key, first_x)["params"]

    tx = make_optimizer(config.optimizer, num_training_steps, epochs)

    loss_fn = make_quantile_loss_fn(config.model.quantiles, dtype=compute_dtype)

    if dynamic_scale == "auto" and compute_dtype == jnp.float16:
        dynamic_scale = DynamicScale()

    if early_stopping == "auto":
        early_stopping = EarlyStopping(best_metric=999, min_delta=0.1, patience=100)

    training_state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
        dynamic_scale=dynamic_scale,
        early_stopping=early_stopping,
    )

    if tensorboard_logdir is None:
        tensorboard_logdir = "tensorboard"

    if isinstance(hooks, Callable):
        hooks = hooks(num_training_steps)
    elif hooks is None:
        tag = make_timestamp_tag()
        tensorboard_logdir = f"{tensorboard_logdir}/{tag}"

        hooks = make_training_hooks(
            num_training_steps, epochs, logdir=tensorboard_logdir, profile=profile, save_path=save_path
        )

    def make_dataset_generator(
        ds: tf.data.Dataset,
    ) -> Callable[[], Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]]:
        def generator():
            for x, y in ds.as_numpy_iterator():
                yield jnp.asarray(x, dtype=compute_dtype), jnp.asarray(y, dtype=compute_dtype)

        return generator

    if multi_device:
        (training_metrics, validation_metrics), params = flax_utils.fit_multi_device(
            training_state=training_state,
            training_dataset_factory=make_dataset_generator(training_dataset),
            validation_dataset_factory=make_dataset_generator(validation_dataset),
            metrics_container_type=MetricContainer,
            training_step_func=multi_device_train_step,
            validation_step_func=multi_device_validation_step,
            epochs=epochs,
            hooks=hooks,
            num_training_steps=num_training_steps,
            prefetch_buffer_size=prefetch_buffer_size,
            verbose=verbose,
        )
    else:
        (training_metrics, validation_metrics), params = flax_utils.fit_single_device(
            training_state=training_state,
            training_dataset_factory=make_dataset_generator(training_dataset),
            validation_dataset_factory=make_dataset_generator(validation_dataset),
            metrics_container_type=MetricContainer,
            training_step_func=single_device_train_step,
            validation_step_func=single_device_validation_step,
            epochs=epochs,
            hooks=hooks,
            num_training_steps=num_training_steps,
            verbose=verbose,
        )

    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")
    return (training_metrics, validation_metrics), params
