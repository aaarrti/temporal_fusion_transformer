from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar
from functools import partial


@partial(jax.jit, static_argnums=(2,))
def quantile_loss(
    y_true: Float[Array, "batch time_steps n"],
    y_pred: Float[Array, "batch time_steps n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Array, "batch"]:
    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)

    ql = jax.vmap(jax.vmap(_quantile_loss, in_axes=[0, 0, None]), in_axes=[0, 0, None])(
        y_true, y_pred, quantiles
    )

    # Average over time-steps.
    return jnp.mean(ql, axis=-1)


@partial(jax.jit, static_argnums=(2,))
def _quantile_loss(
    y_true: Float[Array, "n"],
    y_pred: Float[Array, "n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Scalar]:
    y_pred = jnp.reshape(y_pred, (-1, quantiles.shape[0]))
    prediction_underflow = y_true[..., jnp.newaxis] - y_pred
    # Sum over quantiles and outputs.
    return jnp.sum(
        jnp.add(
            quantiles * jnp.maximum(prediction_underflow, 0),
            (1 - quantiles) * jnp.maximum(-prediction_underflow, 0),
        ),
    )


@jax.jit
def quantile_rmse(
    y_true: Float[Array, "batch time_steps n"],
    y_pred: Float[Array, "batch time_steps n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Array, "batch"]:
    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)


@partial(jax.jit, static_argnums=(2,))
def _quantile_rmse(
    y_true: Float[Array, "n"],
    y_pred: Float[Array, "n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Scalar]:
    pass
