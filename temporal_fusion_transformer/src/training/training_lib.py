from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    Protocol,
    Tuple,
    TypeVar,
    TypedDict,
)
import sys
from dataclasses import dataclass

import jax
import optax
from flax import jax_utils
from flax.training.common_utils import shard_prng_key
from flax import linen as nn
from absl_extra.flax_utils import ParamReplication
from flax.core.frozen_dict import FrozenDict
from flax.struct import field
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jax import tree_util
from jax.random import KeyArray
from jaxtyping import Array, Float, Scalar, jaxtyped

from temporal_fusion_transformer.src.modeling.tft_layers import InputStruct
from temporal_fusion_transformer.src.training.metrics import MetricContainer


if TYPE_CHECKING:
    import tensorflow as tf
    from temporal_fusion_transformer.src.modeling.loss_fn import QuantileLossFn
    from flax.linen.module import CollectionFilter

    from temporal_fusion_transformer.src.config_dict import OptimizerConfig

    class PRNGCollection(TypedDict):
        dropout: KeyArray
        lstm: KeyArray

    class ApplyFunc(Protocol):
        @jaxtyped
        def __call__(
            self,
            params: Mapping[str, ...],
            x: Float[Array, "batch time n"],
            training: bool = False,
            *,
            rngs: PRNGCollection | None = None,
            mutable: CollectionFilter = False,
            capture_intermediates: bool | Callable[[nn.Module, str], bool] = False,
        ) -> Float[Array, "batch time n q"]:
            ...


if sys.version_info >= (3, 10):
    dc_kw = dict(slots=True)
else:
    dc_kw = dict()


@dataclass(frozen=True, **dc_kw)
class EarlyStoppingConfig:
    best_metric: int
    min_delta: float
    patience: int

    @staticmethod
    def default() -> EarlyStoppingConfig:
        return EarlyStoppingConfig(best_metric=999, min_delta=0.1, patience=100)


class TrainStateContainer(TrainState):
    apply_fn: ApplyFunc = field(pytree_node=False)
    loss_fn: QuantileLossFn = field(pytree_node=False)
    rngs: PRNGCollection
    early_stopping: EarlyStopping | None = None
    dynamic_scale: DynamicScale | None = None


@partial(jax.jit, donate_argnums=[0])
def train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    rngs = tree_util.tree_map(lambda k: jax.random.fold_in(key=k, data=state.step), state.rngs)

    def loss_fn(params: FrozenDict) -> Float[Scalar]:
        # pass training=True as positional args, since flax.nn.jit does not support kwargs.
        y = state.apply_fn({"params": params}, x_batch, True, rngs=rngs)
        y_loss = state.loss_fn(y_batch, y)
        # Sum over quantiles, average over batch entries.
        return jnp.mean(jnp.sum(y_loss, axis=-1))

    if state.dynamic_scale is not None:
        # loss scaling logic is taken from https://github.com/google/flax/blob/main/examples/wmt/train.py#L177
        dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(loss_fn)(
            state.params
        )
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


@jax.jit
def validation_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y).mean()
    metrics = MetricContainer.single_from_model_output(loss=loss)
    return metrics


@partial(jax.pmap, axis_name="i", donate_argnums=[0])
def distributed_train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    rngs = tree_util.tree_map(lambda k: jax.random.fold_in(key=k, data=state.step), state.rngs)

    def loss_fn(params: FrozenDict) -> float:
        y = state.apply_fn({"params": params}, x_batch, True, rngs=rngs)
        y_loss = state.loss_fn(y_batch, y)
        # Sum over quantiles, average over batch entries.
        return jnp.mean(jnp.sum(y_loss, axis=-1))

    if state.dynamic_scale is not None:
        dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(
            loss_fn, axis_name="i"
        )(state.params)
        state = state.replace(dynamic_scale=dynamic_scale)
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


@partial(jax.pmap, axis_name="i")
def distributed_validation_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y)
    loss = lax.pmean(loss, axis_name="i")
    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    return metrics


def load_dataset(
    data_dir: str,
    batch_size: int,
    prng_seed: int,
    num_encoder_steps: int,
    shuffle_buffer_size: int = 1024,
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
    config: OptimizerConfig,
    num_training_steps: int,
) -> optax.GradientTransformation:
    """

    Parameters
    ----------
    config
    num_training_steps:
        Total number of training steps (not per epoch!).

    Returns
    -------

    """
    learning_rate = optax.cosine_decay_schedule(
        init_value=config.init_lr,
        decay_steps=int(num_training_steps * config.decay_steps),
        alpha=config.alpha,
    )
    tx = optax.lion(learning_rate)
    if config.mechanize:
        tx = optax.contrib.mechanize(tx)

    if config.clipnorm != 0:
        tx = optax.chain(optax.adaptive_grad_clip(config.clipnorm), tx)

    if config.ema != 0:
        tx = optax.chain(optax.ema(config.ema), tx)

    return tx


TX = TypeVar("TX", bound=optax.OptState)


def restore_optimizer_state(opt_state: TX, restored: Mapping[str, ...]) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(
        tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
    )


def make_param_replication() -> ParamReplication:
    def replicate(ts: TrainStateContainer):
        ts2 = jax_utils.replicate(ts)
        rngs2 = tree_util.tree_map(shard_prng_key, ts.rngs)
        return ts2.replace(rngs=rngs2)

    return ParamReplication(replicate=replicate, un_replicate=jax_utils.unreplicate)


def make_early_stopping(config: EarlyStoppingConfig) -> EarlyStopping:
    return EarlyStopping(
        best_metric=config.best_metric, min_delta=config.min_delta, patience=config.patience
    )


if TYPE_CHECKING:
    TrainFn = Callable[
        [TrainStateContainer, Float[Array, "batch time n"], Float[Array, "batch time n"]],
        Tuple[TrainStateContainer, MetricContainer],
    ]
    ValidationFn = Callable[
        [TrainFn, Float[Array, "batch time n"], Float[Array, "batch time n"]], MetricContainer
    ]
