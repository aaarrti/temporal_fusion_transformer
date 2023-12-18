from __future__ import annotations

from collections.abc import Sequence

from jax import Array
from keras import mixed_precision, ops
from keras.src.losses import LossFunctionWrapper

newaxis = None


class QuantileLoss(LossFunctionWrapper):
    def __init__(self, quantiles: Sequence[float], **kwargs):
        quantiles = ops.cast(quantiles, mixed_precision.dtype_policy().compute_dtype)
        super().__init__(**kwargs, fn=quantile_loss, tau=quantiles)


class PinballLoss(LossFunctionWrapper):
    def __init__(self, tau: float, **kwargs):
        super().__init__(**kwargs, fn=pinball_loss, tau=tau)


def pinball_loss(y_true: Array, y_pred: Array, tau: float) -> Array:
    tau = ops.cast(tau, y_pred.dtype)
    y_true = ops.cast(y_true, y_pred.dtype)

    prediction_underflow = y_true - y_pred
    q_loss = tau * ops.maximum(prediction_underflow, 0.0) + (1.0 - tau) * ops.maximum(
        -prediction_underflow, 0.0
    )

    return ops.sum(q_loss, axis=(-1, -2))


def quantile_loss(y_true: Array, y_pred: Array, tau: Array) -> Array:
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
    Returns
    -------
    loss:
        Loss value.
    """

    tau = ops.cast(tau, y_pred.dtype)
    y_true = ops.cast(y_true, y_pred.dtype)

    prediction_underflow = ops.expand_dims(y_true, axis=2) - y_pred

    q_loss = tau * ops.maximum(prediction_underflow, 0.0) + (1.0 - tau) * ops.maximum(
        -prediction_underflow, 0.0
    )
    return ops.sum(q_loss, axis=(-1, -2, -3))
