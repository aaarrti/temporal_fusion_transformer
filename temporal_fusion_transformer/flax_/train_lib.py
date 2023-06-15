from __future__ import annotations

import collections
import functools
import itertools
from typing import (
    TYPE_CHECKING,
    Protocol,
    Tuple,
    Mapping,
    NamedTuple,
    Generator,
    Literal,
)

import flax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from absl import logging
from clu import metrics
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass
from flax.training.train_state import TrainState
from jaxtyping import Array, Float
from keras_pbar import keras_pbar

from temporal_fusion_transformer.flax_.modeling import (
    TemporalFusionTransformer,
    TFTInput,
)
from temporal_fusion_transformer.flax_.quantile_loss import quantile_loss, quantile_rmse
from temporal_fusion_transformer.utils import map_dict

if TYPE_CHECKING:
    import tensorflow as tf


class TrainingConfig(NamedTuple):
    num_training_steps: int
    logdir: str
    checkpoints_dir: int
    epochs: int = 1
    log_frequency: int = -1
    checkpoint_frequency: int = -1
    prng_seed: int = 42
    prefetch_to_device: bool = False
    precision: Literal[
        "p=f32,c=f32,o=f32",
        "p=f32,c=f16,o=f16",
        "p=f16,c=f16,o=f16",
        "p=f32,c=bf16,o=bf16",
        "p=bf16,c=bf16,o=bf16",
    ] = "p=f32,c=f32,o=f32"


def train_tft_model(
    model: TemporalFusionTransformer,
    optimizer: optax.GradientTransformation,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    config: TrainingConfig,
):
    devices = jax.devices()
    logging.info(f"{devices = }")

    if len(devices) == 1:
        weights = run_on_single_device(
            model, optimizer, train_dataset, validation_dataset, config
        )
    else:
        weights = run_on_multiple_devices(
            model, optimizer, train_dataset, validation_dataset, config
        )


class TrainStateContainer(TrainState):
    quantiles: Float[Array, "q"]
    policy: jmp.Policy = flax.struct.field(pytree_node=False)


class NumpyIterator(Protocol):
    def __next__(self) -> Tuple[Mapping[str, np.ndarray, np.ndarray]]:
        ...


@flax.struct.dataclass
class MetricsContainer(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    rmse: metrics.Average.from_output("rmse")


def run_on_single_device(
    model: TemporalFusionTransformer,
    optimizer: optax.GradientTransformation,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    config: TrainingConfig,
) -> FrozenDict:
    @jax.jit
    def train_step(
        state: TrainStateContainer,
        x_batch: TFTInput,
        y_batch: Float[Array, "batch t 1"],
    ) -> Tuple[TrainStateContainer, MetricsContainer]:
        def loss_fn(params: FrozenDict):
            logits = state.apply_fn(x_batch, {"params": params})
            loss = quantile_loss(logits, y_batch, quantiles=state.quantiles)
            return jnp.mean(loss), logits

        (loss_value, logits_value), grad_value = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        state = state.apply_gradients(grads=grad_value)
        train_metrics = MetricsContainer.single_from_model_output(
            loss=loss_value,
            rmse=jnp.mean(
                quantile_rmse(logits_value, y_batch, quantiles=state.quantiles)
            ),
        )
        return state, train_metrics

    @jax.jit
    def validation_step(
        state: TrainStateContainer,
        x_batch: TFTInput,
        y_batch: Float[Array, "batch t 1"],
    ) -> MetricsContainer:
        logits = state.apply_fn(x_batch, {"params": state.params})
        loss = jnp.mean(quantile_loss(logits, y_batch, quantiles=state.quantiles))
        rmse = jnp.mean(quantile_rmse(logits, y_batch, quantiles=state.quantiles))
        return MetricsContainer.single_from_model_output(loss=loss, rmse=rmse)

    if config.prefetch_to_device:
        train_dataset = prefetch_to_device(train_dataset)
        validation_dataset = prefetch_to_device(validation_dataset)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    x, _ = train_dataset.as_numpy_iterator().take(1).next()
    x = TFTInput.from_dict(x)
    m_params = model.init(prng_key, x)

    train_state = TrainStateContainer.create(
        apply_fn=model.apply,
        params=m_params,
        tx=optimizer,
        quantiles=model.quantiles,
        policy=jmp.get_policy(config.precision),
    )

    train_metrics_container = MetricsContainer.empty()
    validation_metrics_container = MetricsContainer.empty()

    for i in range(config.epochs):
        for x, y in keras_pbar(
            train_dataset.as_numpy_iterator(), config.num_training_steps
        ):
            train_state, train_metrics_i = train_step(
                train_state, TFTInput.from_dict(x), jnp.asarray(y)
            )
            train_metrics_container = train_metrics_container.merge(train_metrics_i)

        for x, y in validation_dataset.as_numpy_iterator():
            validation_metrics_i = validation_step(
                train_state, TFTInput.from_dict(x), jnp.asarray(y)
            )
            validation_metrics_container = validation_metrics_container.merge(
                validation_metrics_i
            )

        train_details = map_dict(
            train_metrics_container.compute(), lambda k: f"train_{k}"
        )
        validation_details = map_dict(
            validation_metrics_container.compute(), lambda k: f"validation_{k}"
        )
        logging.info(
            f"Epoch {i + 1}/{config.epochs}, {train_details}, {validation_details}"
        )


def run_on_multiple_devices(
    model: TemporalFusionTransformer,
    optimizer: optax.GradientTransformation,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    config: TrainingConfig,
) -> FrozenDict:
    @functools.partial(jax.pmap, axis_name="i", in_axes=(None, 0, 0))
    def train_step(
        state: TrainStateContainer,
        x_batch: TFTInput,
        y_batch: Float[Array, "batch t 1"],
    ) -> Tuple[TrainStateContainer, MetricsContainer]:
        def loss_fn(params: FrozenDict):
            logits = state.apply_fn(x_batch, {"params": params})
            loss = quantile_loss(logits, y_batch, quantiles=state.quantiles)
            return jnp.mean(loss), logits

        (loss_value, logits_value), grad_value = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        grad_value = jax.lax.pmean(grad_value, axis_name="i")
        state = state.apply_gradients(grads=grad_value)
        train_metrics = MetricsContainer.single_from_model_output(
            loss=loss_value,
            rmse=jnp.mean(
                quantile_rmse(logits_value, y_batch, quantiles=state.quantiles)
            ),
        )
        return state, train_metrics

    @functools.partial(jax.pmap, axis_name="i", in_axes=(None, 0, 0))
    def validation_step(
        state: TrainStateContainer,
        x_batch: TFTInput,
        y_batch: Float[Array, "batch t 1"],
    ) -> MetricsContainer:
        logits = state.apply_fn(x_batch, {"params": state.params})
        loss = jnp.mean(quantile_loss(logits, y_batch, quantiles=state.quantiles))
        rmse = jnp.mean(quantile_rmse(logits, y_batch, quantiles=state.quantiles))
        return MetricsContainer.single_from_model_output(loss=loss, rmse=rmse)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    x, _ = train_dataset.as_numpy_iterator().take(1).next()
    x = TFTInput.from_dict(x)
    m_params = model.init(prng_key, x)

    train_state = TrainStateContainer.create(
        apply_fn=model.apply,
        params=m_params,
        tx=optimizer,
        quantiles=model.quantiles,
    )

    train_metrics_container = MetricsContainer.empty()
    validation_metrics_container = MetricsContainer.empty()

    for i in range(config.epochs):
        for x, y in keras_pbar(
            train_dataset.as_numpy_iterator(), config.num_training_steps
        ):
            train_state, train_metrics_i = train_step(
                train_state, TFTInput.from_dict(x), jnp.asarray(y)
            )
            train_metrics_i = jnp.mean(train_metrics_i)
            train_metrics_container = train_metrics_container.merge(train_metrics_i)

        for x, y in validation_dataset.as_numpy_iterator():
            validation_metrics_i = validation_step(
                train_state, TFTInput.from_dict(x), jnp.asarray(y)
            )
            validation_metrics_i = jnp.mean(validation_metrics_i)
            validation_metrics_container = validation_metrics_container.merge(
                validation_metrics_i
            )

        train_details = map_dict(
            train_metrics_container.compute(), lambda k: f"train_{k}"
        )
        validation_details = map_dict(
            validation_metrics_container.compute(), lambda k: f"validation_{k}"
        )
        logging.info(
            f"Epoch {i + 1}/{config.epochs}, {train_details}, {validation_details}"
        )


def prefetch_to_device(
    iterator: NumpyIterator[Tuple[Mapping[str, np.ndarray], np.ndarray]],
    size: int = 2,
    device: jax.Device | None = None,
) -> Generator[Tuple[TFTInput, Float[Array, "batch t 1"]], None, None]:
    queue = collections.deque()
    if device is None:
        device = jax.devices()[0]

    for x, y in itertools.islice(iterator, size):
        queue.append((jax.device_put(TFTInput.from_dict(x), jax.device_put(y, device))))
    while queue:
        yield queue.pop()
