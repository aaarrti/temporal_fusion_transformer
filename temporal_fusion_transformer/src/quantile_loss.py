from __future__ import annotations

import functools
from typing import Callable, Sequence, Tuple, TypeAlias

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, jaxtyped

QuantileLossFn: TypeAlias = Callable[
    [Float[Array, "batch time n"], Float[Array, "batch time n*q"]], Float[Array, "batch"]
]


def make_quantile_loss_fn(quantiles: Sequence[float]) -> QuantileLossFn:
    return jax.vmap(Partial(quantile_loss, quantiles=tuple(quantiles)))


@jaxtyped
@functools.partial(jax.jit, static_argnames=["quantiles"])
def quantile_loss(
    y_true: Float[Array, "time n"],
    y_pred: Float[Array, "time n*quantiles"],
    quantiles: Tuple[float],
) -> Float[Array, "batch"]:
    """
    Parameters
    ----------
    y_true:
        Targets
    y_pred:
        Predictions
    quantiles:
        Quantile to use for loss calculations (between 0 & 1)

    Returns
    -------
    loss:
        Loss value.
    """
    quantiles = jnp.asarray(quantiles)
    # jax.debug.print("y_true -> {}, y_pred -> {}", y_true, y_pred)
    prediction_underflow = y_true - y_pred

    over_estimation_error = quantiles * jnp.maximum(prediction_underflow, 0)
    under_estimation_error = (1 - quantiles) * jnp.maximum(-prediction_underflow, 0)

    # Sum over quantiles and outputs * average over time-steps.
    return jnp.mean(jnp.sum(over_estimation_error + under_estimation_error, axis=-1))
