from __future__ import annotations

from typing import List
import tensorflow as tf
from keras.utils.tf_utils import can_jit_compile
from keras.losses import LossFunctionWrapper


class QuantileLoss(LossFunctionWrapper):
    """Computes the combined quantile loss for prespecified quantiles.

    Attributes:
      quantiles: Quantiles to compute losses
    """

    def __init__(self, quantiles: List[int]):
        """Initializes computer with quantiles for loss calculations.

        Args:
          quantiles: Quantiles to use for computations.
        """
        super().__init__()
        self.quantiles = quantiles

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Returns quantile loss for specified quantiles."""
        quantiles_used = set(self.quantiles)

        loss = 0.0
        for i, quantile in enumerate(valid_quantiles):
            if quantile in quantiles_used:
                loss += utils.tensorflow_quantile_loss(
                    a[Ellipsis, output_size * i : output_size * (i + 1)],
                    b[Ellipsis, output_size * i : output_size * (i + 1)],
                    quantile,
                )
        return loss


def quantile_loss(y: tf.Tensor, y_pred: tf.Tensor, quantile: float) -> tf.Tensor:
    """

    Parameters
    ----------
    y:
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
    if quantile < 0 or quantile > 1:
        raise ValueError(
            f"Illegal quantile value={quantile}! Values should be between 0 and 1."
        )

    return _quantile_loss(y, y_pred, tf.constant(quantile))


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def _quantile_loss(y: tf.Tensor, y_pred: tf.Tensor, quantile: tf.Tensor) -> tf.Tensor:
    prediction_underflow = y - y_pred
    positive_prediction_underflow = tf.maximum(prediction_underflow, 0.0)
    negative_prediction_underflow = tf.maximum(-prediction_underflow, 0.0)
    q_loss = (
        quantile * positive_prediction_underflow
        + (1.0 - quantile) * negative_prediction_underflow
    )
    return tf.reduce_sum(q_loss, axis=-1)
