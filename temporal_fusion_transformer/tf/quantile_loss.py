from __future__ import annotations

from typing import Sequence, Dict

import tensorflow as tf
from keras.losses import Loss
from keras.utils.tf_utils import can_jit_compile
from jaxtyping import Float


class QuantileLoss(Loss):
    """
    Computes the combined quantile loss for specified quantiles.

    References:
        - https://arxiv.org/abs/1912.09363

    """

    def __init__(
        self,
        quantiles: Sequence[int],
        output_size: int = 1,
        **kwargs,
    ):
        """

        Parameters
        ----------
        quantiles:
            Quantiles to compute losses
        output_size:
        kwargs

        """
        super().__init__(**kwargs)
        for quantile in quantiles:
            if quantile < 0 or quantile > 1:
                raise ValueError(
                    f"Illegal quantile value={quantile}! Values should be between 0 and 1."
                )
        self.quantiles = tf.constant(quantiles)
        self.output_size = tf.constant(output_size)
        self.n_quantiles = tf.constant(len(quantiles))

    def call(
        self,
        y_true: Float[tf.Tensor, "batch time_steps n"],
        y_pred: Float[tf.Tensor, "batch time_steps n*q"],
    ) -> Float[tf.Tensor, "batch time_steps"]:
        """

        Parameters
        ----------
        y_true:
            Targets of shape (batch, time_steps, 1).
        y_pred:
            Predictions of shape (batch, time_steps, quantiles).

        Returns
        -------

        """

        y_true = tf.cast(y_true, y_pred.dtype)
        quantiles = tf.cast(self.quantiles, y_pred.dtype)
        tf.debugging.assert_rank(y_true, 3)

        # loss = tf.TensorArray(
        #    size=self.n_quantiles, dtype=y_pred.dtype, clear_after_read=True
        # )
        #
        # for i in tf.range(self.n_quantiles):
        #    indexes = tf.range(i * self.output_size, (i + 1) * self.output_size)
        #    q_loss = quantile_loss(
        #        y_true, tf.gather(y_pred, indexes, axis=-1), quantiles[i]
        #    )
        #    loss = loss.write(i, q_loss)
        # FIXME
        loss = [
            quantile_loss(y_true, y_pred[..., 0:1], quantiles[0]),
            quantile_loss(y_true, y_pred[..., 1:2], quantiles[1]),
            quantile_loss(y_true, y_pred[..., 2:3], quantiles[1]),
        ]
        loss = tf.reduce_sum(loss, axis=0)
        return loss

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "quantiles": list(self.quantiles),
                "output_size": int(self.output_size),
                "accumulate_in_tensor_array": self.accumulate_in_tensor_array,
            }
        )
        return config


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def quantile_loss(
    y_true: Float[tf.Tensor, "batch time_steps n"],
    y_pred: Float[tf.Tensor, "batch time_steps n"],
    quantile: Float[tf.Tensor, "1"],
) -> Float[tf.Tensor, "batch time_steps"]:
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
    positive_prediction_underflow = tf.maximum(prediction_underflow, 0)
    negative_prediction_underflow = tf.maximum(-prediction_underflow, 0)
    q_loss = tf.add(
        quantile * positive_prediction_underflow,
        (1 - quantile) * negative_prediction_underflow,
    )
    return tf.reduce_sum(q_loss, axis=-2)
