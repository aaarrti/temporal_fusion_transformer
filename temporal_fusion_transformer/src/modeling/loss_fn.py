from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.lib_types import ComputeDtype, LossFn


def make_quantile_loss_fn(quantiles: Sequence[float], dtype: ComputeDtype = jnp.float32) -> LossFn:
    """

    Parameters
    ----------
    quantiles
    dtype

    Returns
    -------

    fn:
        Function which take 3D and 4D array, and outputs 2D array (batch, quantiles)

    """
    return Partial(
        quantile_loss,
        quantiles=tuple(quantiles),
        dtype=dtype,
    )


@functools.partial(
    jax.jit,
    # Works on GPU, but fails on Kaggle TPU ¯\_(ツ)_/¯
    # static_argnums=[2, 3],
    # static_argnames=["tau", "dtype"],
    static_argnums=[3],
    static_argnames=["dtype"],
    inline=True,
)
def pinball_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    tau: float,
    dtype: ComputeDtype = jnp.float32,
):
    """
    Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    References:
        - https://en.wikipedia.org/wiki/Quantile_regression
        - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/quantiles.py



    Parameters
    ----------
    y_true:
        3D float (batch time n)
    y_pred:
        3D float (batch time n)
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


@functools.partial(
    jax.jit,
    static_argnums=[2, 3],
    static_argnames=["quantiles", "dtype"],
    inline=True,
)
def quantile_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    quantiles: Tuple[float, ...],
    dtype: ComputeDtype = jnp.float32,
):
    """
    Compute pinball loss for different quantiles stacked over last axis.

    Parameters
    ----------
    y_true:
        3D float (batch time n)
    y_pred:
        4D float (batch time n q)
    quantiles
    dtype

    Returns
    -------

    loss:
        Error for different quantiles stacked over last axis.

    """
    quantiles = jnp.asarray(quantiles, dtype)
    q_loss = jax.vmap(pinball_loss, in_axes=[None, -1, -1, None], out_axes=-1)(y_true, y_pred, quantiles, dtype)
    return jnp.sum(q_loss, axis=-1)
