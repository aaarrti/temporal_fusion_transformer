from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Protocol, Tuple, TypeVar

import jax
import optax
from absl_extra.typing_utils import ParamSpec
from flax.core.frozen_dict import FrozenDict
from flax.struct import field
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jax import tree_util
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from ml_collections import ConfigDict

from temporal_fusion_transformer.src.loss_fn import QuantileLossFn
from temporal_fusion_transformer.src.metrics import MetricContainer
from temporal_fusion_transformer.src.tft_layers import InputStruct

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

if TYPE_CHECKING:
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import OptimizerConfig


class ApplyFunc(Protocol):
    def __call__(
        self,
        params: Mapping[str, FrozenDict],
        x: Float[Array, "batch time n"],
        training: bool = False,
        *,
        rngs: Mapping[str, PRNGKeyArray] | None = None,
    ) -> Float[Array, "batch time n"]:
        ...


class TrainStateContainer(TrainState):
    apply_fn: ApplyFunc = field(pytree_node=False)
    loss_fn: QuantileLossFn = field(pytree_node=False)
    dropout_key: PRNGKeyArray
    dynamic_scale: DynamicScale | None = None
    early_stopping: EarlyStopping | None = None


StateAndMetrics = Tuple[TrainStateContainer, MetricContainer]
StepFn = Callable[[TrainStateContainer, InputStruct, Float[Array, "batch time n"]], StateAndMetrics]
TrainFn = Callable[[TrainStateContainer, InputStruct, Float[Array, "batch time n"]], StateAndMetrics]
ValidationFn = Callable[[TrainStateContainer, InputStruct, Float[Array, "batch time n"]], MetricContainer]


class SingleDevice(SimpleNamespace):
    @staticmethod
    def train_step(
        state: TrainStateContainer,
        x_batch: InputStruct,
        y_batch: Float[Array, "batch time n"],
    ) -> StateAndMetrics:
        dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

        def loss_fn(params: FrozenDict) -> Float[Scalar]:
            # pass training=True as positional args, since flax.nn.jit does not support kwargs.
            y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
            y_loss = state.loss_fn(y_batch, y)
            return jnp.sum(y_loss)

        if state.dynamic_scale is not None:
            # loss scaling logic is taken from https://github.com/google/flax/blob/main/examples/wmt/train.py#L177
            dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(loss_fn)(state.params)
            state.replace(dynamic_scale=dynamic_scale)
            state = state.apply_gradients(grads=grads)
        else:
            dynamic_scale, is_fin = None, None
            loss, grads = jax.value_and_grad(loss_fn)(state.params)

        state = state.apply_gradients(grads=grads)
        if state.dynamic_scale is not None:
            select_fn = tree_util.Partial(jnp.where, is_fin)
            state = state.replace(
                opt_state=jax.tree_util.tree_map(select_fn, state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(select_fn, state.params, state.params),
            )

        metrics = MetricContainer.single_from_model_output(loss=loss)
        return state, metrics

    @staticmethod
    def validation_step(
        state: TrainStateContainer,
        x_batch: InputStruct,
        y_batch: Float[Array, "batch time n"],
    ) -> MetricContainer:
        y = state.apply_fn({"params": state.params}, x_batch)
        loss = state.loss_fn(y_batch, y).mean()
        metrics = MetricContainer.single_from_model_output(loss=loss)
        return metrics

    @staticmethod
    def make_train_step_fn() -> TrainFn:
        return jax.jit(SingleDevice.train_step)

    @staticmethod
    def make_validation_step_fn() -> ValidationFn:
        return jax.jit(SingleDevice.validation_step)


class MultiDevice(SimpleNamespace):
    @staticmethod
    def train_step(
        state: TrainStateContainer,
        x_batch: InputStruct,
        y_batch: Float[Array, "batch time n"],
    ) -> Tuple[TrainStateContainer, MetricContainer]:
        dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

        def loss_fn(params: FrozenDict) -> float:
            y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
            y_loss = state.loss_fn(y_batch, y)
            return jnp.sum(y_loss)

        if state.dynamic_scale is not None:
            dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(loss_fn, axis_name="i")(
                state.params
            )
            state.replace(dynamic_scale=dynamic_scale)
        else:
            dynamic_scale, is_fin = None, None
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = lax.pmean(grads, axis_name="i")

        state = state.apply_gradients(grads=grads)
        if state.dynamic_scale is not None:
            select_fn = tree_util.Partial(jnp.where, is_fin)
            state = state.replace(
                opt_state=jax.tree_util.tree_map(select_fn, state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(select_fn, state.params, state.params),
            )

        metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
        return state, metrics

    @staticmethod
    def validation_step(
        state: TrainStateContainer,
        x_batch: InputStruct,
        y_batch: Float[Array, "batch time n"],
    ) -> MetricContainer:
        y = state.apply_fn({"params": state.params}, x_batch)
        loss = state.loss_fn(y_batch, y)
        loss = lax.pmean(loss, axis_name="i")
        metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
        return metrics

    @staticmethod
    def make_train_step_fn() -> TrainFn:
        return jax.pmap(MultiDevice.train_step, axis_name="i")

    @staticmethod
    def make_validation_step_fn() -> ValidationFn:
        return jax.pmap(MultiDevice.validation_step, axis_name="i")


def load_dataset(
    data_dir: str,
    batch_size: int,
    prng_seed: int,
    num_encoder_steps: int,
    shuffle_buffer_size: int = 2048,
    dtype=jnp.float32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """

    Parameters
    ----------
    data_dir
    batch_size
    shuffle_buffer_size:
        If set to None, will do a full-reshuffle.
    prng_seed
    dtype
    num_encoder_steps:
        Number of time steps to consider as past. Those steps will be discarded from y_batch.

    Returns
    -------

    """
    import tensorflow as tf

    tf_dtype = tf.dtypes.as_dtype(dtype)

    def downcast_input(x, y):
        return tf.cast(x, tf_dtype), tf.cast(y, tf_dtype)

    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{split}", compression="GZIP")
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, seed=prng_seed, reshuffle_each_iteration=True)
            .map(downcast_input)
            .map(lambda x, y: (x, y[:, num_encoder_steps:]))
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")
    return training_ds, validation_ds


def make_optimizer(
    config: OptimizerConfig | ConfigDict, num_training_steps: int, epochs: int
) -> optax.GradientTransformation:
    learning_rate = config.learning_rate
    if config.decay_steps != 0:
        decay_steps = num_training_steps * epochs * config.decay_steps
        learning_rate = optax.cosine_decay_schedule(learning_rate, decay_steps, config.decay_alpha)
    tx = optax.adam(learning_rate)
    if config.clipnorm != 0:
        tx = optax.chain(optax.adaptive_grad_clip(config.clipnorm), tx)

    if config.ema != 0:
        tx = optax.chain(tx, optax.ema(config.ema))

    return tx


def restore_optimizer_state(opt_state: T, restored: Mapping[str, ...]) -> T:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored))
