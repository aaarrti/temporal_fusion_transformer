from __future__ import annotations

from datetime import datetime
from typing import Literal, Tuple, Generator, Callable

import jax
import tensorflow as tf
from absl import logging
from absl_extra import flax_utils
from absl_extra.typing_utils import ParamSpec
from flax.training.dynamic_scale import DynamicScale
from jax import numpy as jnp

from temporal_fusion_transformer.src.config_dict import ConfigDict
from temporal_fusion_transformer.src.quantile_loss import make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer, InputStruct
from temporal_fusion_transformer.src.training_lib import (
    MetricContainer,
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


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def train_experiment(
    *,
    data_dir: str,
    batch_size: int,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDict,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    device_type: Literal["cpu", "gpu", "tpu"] = "tpu",
    prefetch_buffer_size: int = 2,
    dynamic_scale: DynamicScale | None = None,
    verbose: bool = True,
    hooks: flax_utils.TrainingHooks | Callable[[int], flax_utils.TrainingHooks] | None = None,
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
    )
    return train(
        data=data,
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
        hooks=hooks
    )


def train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    epochs: int = 1,
    batch_size: int | None = None,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    device_type: Literal["gpu", "tpu"] = "gpu",
    dynamic_scale: DynamicScale | None | Literal["auto"] = None,
    prefetch_buffer_size: int = 2,
    hooks: flax_utils.TrainingHooks | Callable[[int], flax_utils.TrainingHooks] | None = None,
    verbose: bool = True,
) -> flax_utils.MetricsAndParams:
    """

    Parameters
    ----------
    data:
        Tuple of training & validation tf.data.Dataset's
    config:
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
        Number of element to prefetch to the GPU.
    hooks:
        Custom training hooks.
    dynamic_scale:
        Custom loss scale to use for mixed float16 training.
    device_type:
        Relevant only for multi-device setting.

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
        first_x = first_x[:batch_size]

    model = TemporalFusionTransformer.from_config_dict(config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    params = model.init(params_key, first_x)["params"]

    tx = make_optimizer(config.optimizer, num_training_steps, epochs)

    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles, dtype=compute_dtype)

    if dynamic_scale == "auto" and compute_dtype == jnp.float16:
        dynamic_scale = DynamicScale()

    state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
        dynamic_scale=dynamic_scale,
    )
    
    if isinstance(hooks, Callable):
        hooks = hooks(num_training_steps)
    elif hooks is None:
        tag = make_timestamp_tag()
        tensorboard_log_dir = f"tensorboard/{tag}"

        hooks = make_training_hooks(
            num_training_steps,
            epochs,
            logdir=tensorboard_log_dir,
            checkpoint_directory="checkpoints",
            checkpoint_frequency=5,
            log_frequency=(500, 100)
        )
    
    def make_dataset_generator(ds: tf.data.Dataset) -> Callable[[], Generator[Tuple[InputStruct, jnp.ndarray], None, None]]:
        
        def generator():
            for x, y in ds.as_numpy_iterator():
                yield model.make_input_struct(x), jnp.asarray(y, dtype=compute_dtype)
        
        return generator

    if multi_device:
        (training_metrics, validation_metrics), params = flax_utils.fit_multi_device(
            training_state=state,
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
            training_state=state,
            training_dataset_factory=make_dataset_generator(training_dataset),
            validation_dataset_factory=make_dataset_generator(validation_dataset),
            metrics_container_type=MetricContainer,
            training_step_func=single_device_train_step,
            validation_step_func=single_device_validation_step,
            epochs=epochs,
            hooks=hooks,
            num_training_steps=num_training_steps,
            prefetch_buffer_size=prefetch_buffer_size,
            verbose=verbose,
        )

    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")

    if save_path is None:
        flax_utils.save_as_msgpack(params, save_path)

    return (training_metrics, validation_metrics), params
