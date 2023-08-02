from __future__ import annotations

import functools
import inspect
from typing import Callable, Mapping, Protocol, Tuple

import jax
from clu.metrics import Average, Collection, _ReductionCounter
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass, field
from flax.training.dynamic_scale import DynamicScale
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from temporal_fusion_transformer.src.quantile_loss import QuantileLossFn
from temporal_fusion_transformer.src.tft_layers import ComputeDtype, InputStruct

ValidateFunc = Callable[[Float[Array, "batch time n"], Mapping[str, FrozenDict]], Float[Array, "batch time n"]]


class ApplyFunc(Protocol):
    def __call__(
        self,
        params: Mapping[str, FrozenDict],
        x: Float[Array, "batch time n"],
        *,
        rngs: Mapping[str, PRNGKeyArray] | None = None,
        training: bool = False,
    ) -> Float[Array, "batch time n"]:
        ...


@jaxtyped
@dataclass
class MetricContainer(Collection):
    loss: Average.from_output("loss")

    @classmethod
    def empty(cls) -> MetricContainer:
        return MetricContainer(
            _reduction_counter=_ReductionCounter.empty(),
            **{
                metric_name: metric.empty()
                for metric_name, metric in inspect.get_annotations(cls, eval_str=True).items()
            },
        )

    @classmethod
    def _from_model_output(cls, **kwargs) -> MetricContainer:
        """Creates a `Collection` from model outputs."""
        return MetricContainer(
            _reduction_counter=_ReductionCounter.empty(),
            **{
                metric_name: metric.from_model_output(**kwargs)
                for metric_name, metric in inspect.get_annotations(cls, eval_str=True).items()
            },
        )


@jaxtyped
class TrainStateContainer(TrainState):
    apply_fn: ApplyFunc = field(pytree_node=False)
    loss_fn: QuantileLossFn = field(pytree_node=False)
    dropout_key: PRNGKeyArray
    loss_scale: DynamicScale = field(pytree_node=False)
    compute_dtype: ComputeDtype = field(pytree_node=False)

    @classmethod
    def create(
        cls, *, apply_fn, params, tx, compute_dtype: ComputeDtype = jnp.float32, **kwargs
    ) -> TrainStateContainer:
        if compute_dtype == jnp.float16:
            loss_scale = DynamicScale()
        else:
            loss_scale = None

        return super().create(
            apply_fn=apply_fn, params=params, tx=tx, loss_scale=loss_scale, compute_dtype=compute_dtype, **kwargs
        )


@jaxtyped
@jax.jit
def single_device_train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    x_batch = x_batch.cast_inexact(state.compute_dtype)

    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

    def loss_fn(params: FrozenDict) -> float:
        y = state.apply_fn({"params": params}, x_batch, training=True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.loss_scale is not None:
        _, _, loss, grads = state.loss_scale.value_and_grad(loss_fn)(state.params)
    else:
        loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = MetricContainer.single_from_model_output(loss=loss)
    return state, metrics


@jaxtyped
@jax.jit
def single_device_validation_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    x_batch = x_batch.cast_inexact(state.compute_dtype)
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y).mean()
    metrics = MetricContainer.single_from_model_output(loss=loss)
    return metrics


@jaxtyped
@functools.partial(jax.pmap, axis_name="i")
def multi_device_train_step(
    state: TrainStateContainer,
    x_batch: Float[Array, "batch time n"],
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

    def loss_fn(params: FrozenDict) -> float:
        y = state.apply_fn({"params": params}, x_batch, training=True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.loss_fn is not None:
        _, _, loss, grads = state.loss_scale.value_and_grad(loss_fn, axis_name="i")(state.params)
    else:
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name="i")

    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jaxtyped
@functools.partial(jax.pmap, axis_name="i")
def multi_device_validation_step(
    state: TrainStateContainer,
    x_batch: Float[Array, "batch time n"],
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y)
    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    return metrics
