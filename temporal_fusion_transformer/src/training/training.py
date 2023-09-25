from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Generator, Literal, Tuple

import jax
import numpy as np
from absl import logging
from absl_extra import flax_utils
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp

from temporal_fusion_transformer.src.modeling.loss_fn import make_quantile_loss_fn
from temporal_fusion_transformer.src.modeling.tft_model import (
    make_temporal_fusion_transformer,
)
from temporal_fusion_transformer.src.training.metrics import MetricContainer
from temporal_fusion_transformer.src.training.training_hooks import HooksConfig, make_training_hooks
from temporal_fusion_transformer.src.training.training_lib import (
    TrainStateContainer,
    distributed_train_step,
    distributed_validation_step,
    make_optimizer,
    train_step,
    validation_step,
    make_param_replication,
)

if TYPE_CHECKING:
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import ConfigDict, DatasetConfig
    from temporal_fusion_transformer.src.modeling.tft_layers import ComputeDtype
    from temporal_fusion_transformer.src.training.training_lib import (
        TrainFn,
        ValidationFn,
    )

    HooksT = (
        flax_utils.TrainingHooks
        | Callable[[int], flax_utils.TrainingHooks]
        | Literal["auto"]
        | HooksConfig
        | None
    )
    DynamicScaleT = DynamicScale | None | Literal["auto"]
    EarlyStoppingT = EarlyStopping | None | Literal["auto"]
    DeviceTypeT = Literal["gpu", "tpu"]


def train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    data_config: DatasetConfig,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    dynamic_scale: DynamicScaleT = None,
    hooks: HooksT = "auto",
    verbose: bool = True,
    profile: bool = False,
    early_stopping: EarlyStoppingT = "auto",
    tensorboard_logdir: str | None = "tensorboard",
) -> flax_utils.MetricsAndParams:
    compute_dtype = jnp.float16 if mixed_precision else jnp.float32

    return _train(
        data=data,
        config=config,
        prefetch_buffer_size=0,
        dynamic_scale=dynamic_scale,
        early_stopping=early_stopping,
        hooks=hooks,
        save_path=save_path,
        tensorboard_logdir=tensorboard_logdir,
        compute_dtype=compute_dtype,
        data_config=data_config,
        epochs=epochs,
        verbose=verbose,
        jit_module=jit_module,
        profile=profile,
        tabulate_model=False,
        train_step_fn=train_step,
        validation_step_fn=validation_step,
    )


def train_distributed(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    data_config: DatasetConfig,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    device_type: DeviceTypeT = "gpu",
    dynamic_scale: DynamicScaleT = None,
    prefetch_buffer_size: int = 0,
    hooks: HooksT = "auto",
    verbose: bool = True,
    tensorboard_logdir: str = None,
    profile: bool = False,
    early_stopping: EarlyStoppingT = "auto",
    tabulate_model: bool = False,
) -> flax_utils.MetricsAndParams:
    num_devices = jax.device_count()

    if num_devices == 1:
        raise RuntimeError("Expected multiple JAX devices.")

    if prefetch_buffer_size != 0 and device_type == "tpu":
        logging.warning("`prefetch_buffer_size` must be 0 for TPU")
        prefetch_buffer_size = 0

    if mixed_precision:
        if device_type == "gpu":
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.bfloat16
    else:
        compute_dtype = jnp.float32

    return _train(
        data=data,
        compute_dtype=compute_dtype,
        dynamic_scale=dynamic_scale,
        early_stopping=early_stopping,
        hooks=hooks,
        verbose=verbose,
        prefetch_buffer_size=prefetch_buffer_size,
        config=config,
        data_config=data_config,
        epochs=epochs,
        profile=profile,
        jit_module=jit_module,
        tabulate_model=tabulate_model,
        save_path=save_path,
        tensorboard_logdir=tensorboard_logdir,
        train_step_fn=distributed_train_step,
        validation_step_fn=distributed_validation_step,
    )


def _train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    data_config: DatasetConfig,
    config: ConfigDict,
    hooks: HooksT,
    verbose: bool,
    profile: bool,
    early_stopping: EarlyStoppingT,
    prefetch_buffer_size: int,
    dynamic_scale: DynamicScaleT,
    tabulate_model: bool,
    compute_dtype: ComputeDtype,
    epochs: int,
    jit_module: bool,
    tensorboard_logdir: str | None,
    save_path: str | None,
    train_step_fn: TrainFn,
    validation_step_fn: ValidationFn,
) -> flax_utils.MetricsAndParams:
    device_count = jax.device_count()

    training_dataset, validation_dataset = data

    num_training_steps = int(training_dataset.cardinality())
    first_x: np.ndarray = training_dataset.as_numpy_iterator().next()[0]
    batch_size = first_x.shape[0] // device_count
    first_x = jnp.asarray(first_x[:batch_size], dtype=compute_dtype)

    tx = make_optimizer(config.optimizer, num_training_steps * epochs)

    loss_fn = make_quantile_loss_fn(config.model.quantiles, dtype=compute_dtype)

    if dynamic_scale == "auto" and compute_dtype == jnp.float16:
        dynamic_scale = DynamicScale()

    if early_stopping == "auto":
        early_stopping = EarlyStopping(best_metric=999, min_delta=0.1, patience=100)

    if hooks == "auto":
        if tensorboard_logdir is not None:
            tensorboard_logdir = Path(tensorboard_logdir).joinpath(make_timestamp_tag()).as_posix()
        hooks = make_training_hooks(
            num_training_steps,
            epochs,
            logdir=tensorboard_logdir,
            profile=profile,
            save_path=save_path,
            delete_checkpoints_after_training=True,
        )
    elif isinstance(hooks, HooksConfig):
        hooks = hooks.make_training_hooks(num_training_steps=num_training_steps, epochs=epochs)
    elif isinstance(hooks, Callable):
        hooks = hooks(num_training_steps)

    model = make_temporal_fusion_transformer(
        config, data_config, jit_module=jit_module, dtype=compute_dtype
    )

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key, lstm_key = jax.random.split(prng_key, 3)

    if tabulate_model:
        table = model.tabulate(params_key, first_x)
        logging.info(table)

    params = model.init(params_key, first_x)["params"]

    training_state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        dynamic_scale=dynamic_scale,
        early_stopping=early_stopping,
        rngs={"dropout": dropout_key, "lstm": lstm_key},
    )

    (training_metrics, validation_metrics), params = flax_utils.fit(
        training_state=training_state,
        training_dataset_factory=make_dataset_generator(training_dataset, compute_dtype),
        validation_dataset_factory=make_dataset_generator(validation_dataset, compute_dtype),
        metrics_container_type=MetricContainer,
        training_step_func=train_step_fn,
        validation_step_func=validation_step_fn,
        epochs=epochs,
        hooks=hooks,
        num_training_steps=num_training_steps,
        prefetch_buffer_size=prefetch_buffer_size,
        verbose=verbose,
        param_replication=make_param_replication(),
    )

    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")
    return (training_metrics, validation_metrics), params


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def make_dataset_generator(
    ds: tf.data.Dataset, compute_dtype: jnp.float32 | jnp.float16 | jnp.bfloat16
) -> Callable[[], Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]]:
    def generator():
        for x, y in ds.as_numpy_iterator():
            yield jnp.asarray(x, dtype=compute_dtype), jnp.asarray(y, dtype=compute_dtype)

    return generator
