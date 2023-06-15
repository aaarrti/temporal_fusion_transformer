from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def quantile_loss(
    y_true: Float[Array, "batch time_steps n"],
    y_pred: Float[Array, "batch time_steps n*q"],
    quantiles: Float[Array, "q"],
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

    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)

    batch_size = y_true.shape[0]
    time_steps = y_true.shape[1]
    num_outputs = y_true.shape[2]
    num_quantiles = quantiles.shape[0]

    y_pred = jnp.reshape(y_pred, [batch_size, time_steps, num_outputs, num_quantiles])
    prediction_underflow = y_true[..., jnp.newaxis] - y_pred

    return jnp.mean(
        # Average over time-steps.
        jnp.sum(
            # Sum over quantiles and outputs.
            jnp.add(
                quantiles * jnp.maximum(prediction_underflow, 0),
                (1 - quantiles) * jnp.maximum(-prediction_underflow, 0),
            ),
            axis=[-1, -2],
        ),
        axis=-1,
    )


def quantile_rmse(
    y_true: Float[Array, "batch time_steps n"],
    y_pred: Float[Array, "batch time_steps n*q"],
    quantiles: Float[Array, "q"],
) -> Float[Array, "batch"]:
    y_true = jnp.asarray(y_true, y_pred.dtype)
    quantiles = jnp.asarray(quantiles, y_pred.dtype)

    batch_size = y_true.shape[0]
    time_steps = y_true.shape[1]
    num_outputs = y_true.shape[2]
    num_quantiles = quantiles.shape[0]

    y_pred = jnp.reshape(y_pred, [batch_size, time_steps, num_outputs, num_quantiles])
    # Calculate squared differences, average sum across quantiles.
    squared_diff = jnp.sum(
        jnp.square(y_true[..., jnp.newaxis] - y_pred) * quantiles, axis=-1
    )
    # Average squared differences across time steps and outputs.
    mean_squared_diff = jnp.mean(squared_diff, axis=[-1, -2])
    # Calculate RMSE
    rmse = jnp.sqrt(mean_squared_diff)
    return rmse
