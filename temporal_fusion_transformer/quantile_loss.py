from __future__ import annotations

from typing import List
import tensorflow as tf
from keras.utils.tf_utils import can_jit_compile
from keras.losses import Loss


class QuantileLoss(Loss):
    """
    Computes the combined quantile loss for specified quantiles.

    References:
        - https://arxiv.org/abs/1912.09363

    """

    def __init__(self, quantiles: List[int], output_size: int = 1):
        """

        Parameters
        ----------
        quantiles:
            Quantiles to compute losses
        """
        super().__init__()
        for quantile in quantiles:
            if quantile < 0 or quantile > 1:
                raise ValueError(
                    f"Illegal quantile value={quantile}! Values should be between 0 and 1."
                )
        self.quantiles = quantiles
        self.output_size = output_size

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """

        Parameters
        ----------
        y_true:
            Targets
        y_pred:
            Predictions

        Returns
        -------

        """
        quantiles_used = set(self.quantiles)

        loss = 0.0
        for i, quantile in enumerate(self.quantiles):
            if quantile in quantiles_used:
                loss += quantile_loss(
                    y_true[..., self.output_size * i : self.output_size * (i + 1)],
                    y_pred[Ellipsis, self.output_size * i : self.output_size * (i + 1)],
                    tf.constant(quantile),
                )
        return loss


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def quantile_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, quantile: tf.Tensor
) -> tf.Tensor:
    """
    Parameters
    ----------
    y_true:
        Targets
    y_pred:
        Predictions
    quantile:
        Quantile to use for loss calculations (between 0 & 1)

    Returns
    -------
    loss:
        Loss value.
    """
    prediction_underflow = y_true - y_pred
    positive_prediction_underflow = tf.maximum(prediction_underflow, 0.0)
    negative_prediction_underflow = tf.maximum(-prediction_underflow, 0.0)
    q_loss = (
        quantile * positive_prediction_underflow
        + (1.0 - quantile) * negative_prediction_underflow
    )
    return tf.reduce_sum(q_loss, axis=-1)
