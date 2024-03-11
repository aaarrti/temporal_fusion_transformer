from __future__ import annotations

import jax
from functools import partial
import jax.numpy as jnp
from jax.typing import DTypeLike

_epsilon = jnp.asarray(1e-6)


@partial(jax.jit, static_argnums=[2, 3], static_argnames=["tau", "dtype"])
def quantile_pinball_loss(
    y_true: jax.Array,
    y_pred: jax.Array,
    tau: tuple[float, ...] = (0.1, 0.5, 0.9),
    dtype: DTypeLike = jnp.float32,
) -> jax.Array:
    """

    Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    References:
        - https://en.wikipedia.org/wiki/Quantile_regression
        - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/quantiles.py



    Parameters
    ----------
    y_true:
        4D float (batch time n q)
    y_pred:
        3D float (batch time n)
    tau:
        Quantiles to use for loss calculations with shape (q,), each (between 0 & 1)
        must be a tuple in order for __hash__ to work
    dtype

    Returns
    -------
    loss:
        Loss value with shape (batch_size,)
    """

    with jax.named_scope("quantile_pinball_loss"):
        y_true = y_true.astype(dtype)
        y_true = y_true.astype(dtype)
        q_losses = jax.vmap(
            pinball_loss,
            in_axes=[None, -1, 0],
            out_axes=-1,
        )(y_true, y_pred, jnp.asarray(tau, dtype))
        # sum over quantiles
        return jnp.sum(q_losses, axis=-1)


@partial(jax.jit, inline=True, static_argnums=[2], static_argnames=["tau"])
def pinball_loss(y_true: jax.Array, y_pred: jax.Array, tau: jax.Array) -> jax.Array:
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

    Returns
    -------
    loss:
        Loss value with shape (batch_size,)
    """

    with jax.named_scope("pinball_loss"):
        tau = jnp.asarray(tau, y_pred.dtype)
        error = y_true - y_pred
        under_estimation_error = tau * jnp.maximum(error, 0.0)
        over_estimation_error = (1 - tau) * jnp.maximum(-error, 0.0)
        # average over batch
        return jnp.mean(
            # sum over outputs + time steps
            jnp.sum(over_estimation_error + under_estimation_error, axis=(-1, -2)),
        )
