from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Scalar, jaxtyped

if TYPE_CHECKING:
    QuantileLossFn = Callable[
        [Float[Array, "batch time n"], Float[Array, "batch time n q"]], Float[Array, "batch q"]
    ]


def make_quantile_loss_fn(
    quantiles: Sequence[float], dtype: jnp.inexact = jnp.float32
) -> QuantileLossFn:
    return Partial(
        jax.jit(quantile_loss, static_argnames=["quantiles", "dtype"], donate_argnums=[0, 1]),
        quantiles=tuple(quantiles),
        dtype=dtype,
    )


@jaxtyped
@functools.partial(jax.jit, static_argnums=[2, 3], static_argnames=["tau", "dtype"], inline=True)
def pinball_loss(
    y_true: Float[Array, "batch time n"],
    y_pred: Float[Array, "batch time n"],
    tau: Float[Scalar],
    dtype: jnp.inexact = jnp.float32,
) -> Float[Array, "batch 1"]:
    """
    Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    References:
        - https://en.wikipedia.org/wiki/Quantile_regression
        - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/quantiles.py



    Parameters
    ----------
    y_true:
        Targets
    y_pred:
        Predictions
    tau:
        Quantile to use for loss calculations (between 0 & 1)
    dtype:
        dtype of computation.

    Returns
    -------
    loss:
        Loss value.
    """

    tau = jnp.asarray(tau).astype(dtype)
    y_true = y_true.astype(dtype)
    y_pred = y_pred.astype(dtype)
    prediction_underflow = y_true - y_pred

    over_estimation_error = tau * jnp.maximum(prediction_underflow, 0.0)
    under_estimation_error = (1 - tau) * jnp.maximum(-prediction_underflow, 0.0)

    # Sum over outputs
    error = jnp.sum(over_estimation_error + under_estimation_error, axis=2)
    # Average over time steps
    return jnp.maximum(jnp.mean(error, axis=1), jnp.finfo(dtype).eps)


@jaxtyped
def quantile_loss(
    y_true: Float[Array, "batch time n"],
    y_pred: Float[Array, "batch time n q"],
    quantiles: Float[Array, "q"],
    dtype: jnp.inexact = jnp.float32,
) -> Float[Array, "batch n"]:
    """
    Compute pinball loss for different quantiles stacked over last axis.

    Parameters
    ----------
    y_true
    y_pred
    quantiles
    dtype

    Returns
    -------

    loss:
        Error for different quantiles stacked over last axis.

    """
    quantiles = jnp.asarray(quantiles, dtype)
    return jax.vmap(pinball_loss, in_axes=[None, -1, -1, None], out_axes=-1)(
        y_true, y_pred, quantiles, dtype
    )
