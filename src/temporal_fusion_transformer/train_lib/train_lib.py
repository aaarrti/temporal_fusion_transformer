from __future__ import annotations

from functools import partial
from typing import Protocol, TypedDict, Any, Callable, Iterable, Mapping, TYPE_CHECKING, Unpack
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.training import train_state

from temporal_fusion_transformer.train_lib.loss_fn import quantile_pinball_loss


if TYPE_CHECKING:
    from temporal_fusion_transformer.modeling.model import TftOutputs
    import optax


class ParamCollection(TypedDict):
    params: Mapping[str, Any]


class _TrainStateKwargs(TypedDict):
    prng_key: jax.Array


class ApplyFn(Protocol):
    def __call__(
        self,
        params: ParamCollection,
        x: jax.Array,
        *args,
        rngs: Mapping[str, jax.Array] | None = None,
        training: bool = False,
        capture_intermediates: bool | Callable[[nn.Module, str], bool] = False,
        **kwargs,
    ) -> TftOutputs: ...


@struct.dataclass
class TrainState(train_state.TrainState):
    apply_fn: ApplyFn = struct.field(pytree_node=False)
    prng_key: jax.Array

    @classmethod
    def create(
        cls,
        *,
        apply_fn: ApplyFn,
        params: dict[str, Any],
        tx: optax.GradientTransformation,
        **kwargs: Unpack[_TrainStateKwargs],
    ) -> TrainState:
        return super().create(apply_fn=apply_fn, params=params, tx=tx, **kwargs)


@partial(jax.jit, donate_argnums=[0])
def train_step(state: TrainState, x: jax.Array, y: jax.Array) -> tuple[TrainState, jax.Array]:
    """
    Returns
    -------

    state:
    loss:

    """
    prng_key = jax.random.fold_in(state.prng_key, state.step)

    def loss_fn(params: dict[str, Any]) -> jax.Array:
        logits = state.apply_fn(
            {"params": params}, x, rngs={"dropout": prng_key}, training=True
        ).logits
        return quantile_pinball_loss(y, logits).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads, prng_key=prng_key)
    return state, loss


@jax.jit
def eval_step(tr_st: TrainState, x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Returns
    -------
    metrics: loss
    """
    logits = tr_st.apply_fn({"params": tr_st.params}, x).logits
    return quantile_pinball_loss(y, logits).mean()


def enumerate_batches(
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    prng_key: jax.Array | None = None,
) -> Iterable[tuple[int, jax.Array, jax.Array]]:
    """
    Parameters
    ----------

    x:
    y:
    batch_size:

    prng_key:
        if provider, will shuffle array along 1st axis (you probably shouldn't though).

    """

    if prng_key is not None:
        p = jax.random.permutation(prng_key, jnp.arange(0, len(x)))
        x = jnp.take(x, p, axis=0)
        y = jnp.take(y, p, axis=0)

    for i, batch in enumerate(gen_batches(len(x), batch_size)):
        x_batch = jnp.asarray(x[batch.start : batch.stop])
        y_batch = jnp.asarray(y[batch.start : batch.stop])
        yield i, x_batch, y_batch


def gen_batches(n: int, batch_size: int, *, min_batch_size: int = 0) -> Iterable[slice]:
    # Copied from sklearn
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)
