from __future__ import annotations

import functools
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import AbstractDtype, Array, Float, jaxtyped

QuantileLossFn = Callable[[Float[Array, "batch time n"], Float[Array, "batch time n*q"]], Float[Array, "batch"]]


def make_quantile_loss_fn(quantiles: Sequence[float], dtype=jnp.float32) -> QuantileLossFn:
    return jax.vmap(Partial(quantile_loss, quantiles=tuple(quantiles), dtype=dtype))


@jaxtyped
@functools.partial(jax.jit, static_argnames=["quantiles", "dtype"])
def quantile_loss(
    y_true: Float[Array, "time n"],
    y_pred: Float[Array, "time n*q"],
    quantiles: Tuple[float, ...],
    dtype: AbstractDtype = jnp.float32,
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
    dtype:
        dtype of computation.

    Returns
    -------
    loss:
        Loss value.
    """

    quantiles = jnp.asarray(quantiles).astype(dtype)
    y_true = y_true.astype(dtype)
    y_pred = y_pred.astype(dtype)
    prediction_underflow = y_true - y_pred

    over_estimation_error = quantiles * jnp.maximum(prediction_underflow, 0.0)
    under_estimation_error = (1 - quantiles) * jnp.maximum(-prediction_underflow, 0.0)

    # Sum over quantiles and outputs * average over time-steps.
    return jnp.maximum(jnp.sum(over_estimation_error + under_estimation_error), jnp.finfo(dtype).eps)
