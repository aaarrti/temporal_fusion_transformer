from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar
from functools import partial


@jax.jit
def quantile_loss(
    y_true: Float[Array, "batch time_steps n"],
    y_pred: Float[Array, "batch time_steps n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Array, "batch"]:
    return _quantile_loss(y_true, y_pred, quantiles)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None))
def _quantile_loss(
    y_true: Float[Array, "time_steps n"],
    y_pred: Float[Array, "time_steps n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Scalar]:
    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)

    y_pred = unsqueze_quantiles(y_true, y_pred, quantiles)
    prediction_underflow = y_true[..., tf.newaxis] - y_pred

    return tf.reduce_mean(
        # Average over time-steps.
        tf.reduce_sum(
            # Sum over quantiles and outputs.
            tf.add(
                quantiles * tf.maximum(prediction_underflow, 0),
                (1 - quantiles) * tf.maximum(-prediction_underflow, 0),
            ),
            axis=[-1, -2],
        ),
        axis=-1,
    )


@jax.jit
def quantile_rmse(
    y_true: Float[tf.Tensor, "batch time_steps n"],
    y_pred: Float[tf.Tensor, "batch time_steps n*q"],
    quantiles: Float[tf.Tensor, "q"],
) -> Float[tf.Tensor, "batch"]:
    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)


@partial(jax.jit, static_argnums=(2,))
def _quantile_rmse(
    y_true: Float[Array, "n"],
    y_pred: Float[Array, "n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Scalar]:
    y_pred = unsqueze_quantiles(y_pred, quantiles.shape[0])
    # Calculate squared differences, average sum across quantiles.
    squared_diff = jnp.sum(
        jnp.square(y_true[..., jnp.newaxis] - y_pred) * quantiles, axis=-1
    )
    # Average squared differences across time steps and outputs.
    mean_squared_diff = tf.reduce_mean(squared_diff, axis=[-1, -2])
    # Calculate RMSE
    rmse = tf.sqrt(mean_squared_diff)
    return rmse


@jax.jit
def unsqueze_quantiles(
    y_pred: Float[Array, "batch time_steps n*q"], n_quantiles: Float[Scalar]
) -> Float[Array, "batch time_steps n q"]:
    return jax.vmap(
        jax.vmap(_unsqueze_quantiles, in_axes=(0, None)), in_axes=(0, None)
    )(y_pred, n_quantiles)


@jax.jit
def _unsqueze_quantiles(
    y_pred: Float[Array, "n*q"], n_quantiles: Float[Scalar]
) -> Float[Array, "n q"]:
    return jnp.reshape(y_pred, (-1, n_quantiles))
