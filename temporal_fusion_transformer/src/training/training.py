from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Generator, Tuple

import jax
import numpy as np
from absl import logging
from absl_extra import flax_utils
from flax.training.dynamic_scale import DynamicScale
from jax import numpy as jnp

from temporal_fusion_transformer.src.modeling.loss_fn import make_quantile_loss_fn
from temporal_fusion_transformer.src.modeling.tft_model import (
    make_temporal_fusion_transformer,
)
from temporal_fusion_transformer.src.training.metrics import MetricContainer
from temporal_fusion_transformer.src.training.training_hooks import (
    EarlyStoppingConfig,
    HooksConfig,
    make_training_hooks,
)
from temporal_fusion_transformer.src.training.training_lib import (
    TrainStateContainer,
    distributed_train_step,
    distributed_validation_step,
    make_early_stopping,
    make_optimizer,
    make_param_replication,
    train_step,
    validation_step,
)

if TYPE_CHECKING:
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import ConfigDict, DataConfig
    from temporal_fusion_transformer.src.lib_types import (
        DeviceTypeT,
        DynamicScaleT,
        EarlyStoppingT,
        HooksT,
        TrainingResult,
    )
    from temporal_fusion_transformer.src.modeling.tft_layers import ComputeDtype


def train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    data_config: DataConfig,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    dynamic_scale: DynamicScaleT = None,
    hooks: HooksT = "auto",
    verbose: bool = True,
    early_stopping: EarlyStoppingT = "auto",
) -> TrainingResult:
    compute_dtype = jnp.float16 if mixed_precision else jnp.float32

    return _train(
        data=data,
        config=config,
        prefetch_buffer_size=0,
        dynamic_scale=dynamic_scale,
        early_stopping=early_stopping,
        hooks=hooks,
        compute_dtype=compute_dtype,
        data_config=data_config,
        epochs=epochs,
        verbose=verbose,
        jit_module=jit_module,
        train_step_fn=train_step,
        validation_step_fn=validation_step,
    )


def train_distributed(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    data_config: DataConfig,
    device_type: DeviceTypeT,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    dynamic_scale: DynamicScaleT = None,
    prefetch_buffer_size: int = 0,
    hooks: HooksT = "auto",
    verbose: bool = True,
    early_stopping: EarlyStoppingT = "auto",
) -> TrainingResult:
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
        jit_module=jit_module,
        train_step_fn=distributed_train_step,
        validation_step_fn=distributed_validation_step,
    )


def _train(
    *,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    data_config: DataConfig,
    config: ConfigDict,
    hooks: HooksT,
    verbose: bool,
    early_stopping: EarlyStoppingT,
    prefetch_buffer_size: int,
    dynamic_scale: DynamicScaleT,
    compute_dtype: ComputeDtype,
    epochs: int,
    jit_module: bool,
    train_step_fn: Callable,
    validation_step_fn: Callable,
) -> TrainingResult:
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
        early_stopping = EarlyStoppingConfig()

    if isinstance(early_stopping, EarlyStoppingConfig):
        early_stopping = make_early_stopping(early_stopping)

    if hooks == "auto":
        hooks = HooksConfig()
    if isinstance(hooks, HooksConfig):
        hooks = make_training_hooks(hooks, num_training_steps=num_training_steps, epochs=epochs)

    model = make_temporal_fusion_transformer(config.model, data_config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key, lstm_key = jax.random.split(prng_key, 3)

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
    ds: tf.data.Dataset, compute_dtype: ComputeDtype
) -> Callable[[], Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]]:
    def generator():
        for x, y in ds.as_numpy_iterator():
            yield jnp.asarray(x, dtype=compute_dtype), jnp.asarray(y, dtype=compute_dtype)

    return generator
