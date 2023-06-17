from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from jaxtyping import Float
from keras.losses import LossFunctionWrapper
from keras.metrics import MeanMetricWrapper
from temporal_fusion_transformer.utils import assert_quantile_values, as_tensor, can_jit_compile


class QuantileLoss(LossFunctionWrapper):
    """
    Computes the combined quantile loss for specified quantiles.

    References:
        - https://arxiv.org/abs/1912.09363

    """

    def __init__(
        self,
        quantiles: Sequence[float] | None = None,
        name: str = "quantile_loss",
        **kwargs,
    ):
        assert_quantile_values(quantiles)
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        super().__init__(
            fn=quantile_loss, name=name, **kwargs, quantiles=as_tensor(quantiles)
        )


class QuantileRMSE(MeanMetricWrapper):
    def __init__(
        self,
        quantiles: Sequence[float] | None = None,
        name: str = "quantile_rmse",
        **kwargs,
    ):
        assert_quantile_values(quantiles)
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        super().__init__(
            **kwargs, quantiles=as_tensor(quantiles), name=name, fn=quantile_rmse
        )


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def quantile_loss(
    y_true: Float[tf.Tensor, "batch time_steps n"],
    y_pred: Float[tf.Tensor, "batch time_steps n*q"],
    quantiles: Float[tf.Tensor, "q"],
) -> Float[tf.Tensor, "batch"]:
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

    y_true = tf.cast(y_true, y_pred.dtype)
    quantiles = tf.cast(quantiles, y_pred.dtype)

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


@tf.function(jit_compile=can_jit_compile(True), reduce_retracing=True)
def quantile_rmse(
    y_true: Float[tf.Tensor, "batch time_steps n"],
    y_pred: Float[tf.Tensor, "batch time_steps n*q"],
    quantiles: Float[tf.Tensor, "q"],
) -> Float[tf.Tensor, "batch"]:
    y_true = tf.cast(y_true, y_pred.dtype)
    quantiles = tf.cast(quantiles, y_pred.dtype)
    y_pred = unsqueze_quantiles(y_true, y_pred, quantiles)
    # Calculate squared differences, average sum across quantiles.
    squared_diff = tf.reduce_sum(
        tf.square(y_true[..., tf.newaxis] - y_pred) * quantiles, axis=-1
    )
    # Average squared differences across time steps and outputs.
    mean_squared_diff = tf.reduce_mean(squared_diff, axis=[-1, -2])
    # Calculate RMSE
    rmse = tf.sqrt(mean_squared_diff)
    return rmse


@tf.function(jit_compile=can_jit_compile(True), reduce_retracing=True)
def unsqueze_quantiles(
    y_true: Float[tf.Tensor, "batch time_steps n"],
    y_pred: Float[tf.Tensor, "batch time_steps n*q"],
    quantiles: Float[tf.Tensor, "q"],
) -> Float[tf.Tensor, "batch time_steps n q"]:
    tf.debugging.assert_rank(y_true, 3)
    tf.debugging.assert_rank(y_pred, 3)
    n_quantiles = len(quantiles)
    tf.debugging.assert_shapes(
        [(y_true, ("N", "T", "D")), (y_pred, ("N", "T", f"D*{str(int(n_quantiles))}"))]
    )
    return tf.reshape(y_pred, tf.concat([tf.shape(y_true), [n_quantiles]], axis=0))
