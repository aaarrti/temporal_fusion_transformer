from __future__ import annotations

from collections.abc import Sequence

from jax import Array
from keras_core import mixed_precision, ops
from keras_core.src.losses import LossFunctionWrapper

newaxis = None


class QuantileLoss(LossFunctionWrapper):
    def __init__(self, quantiles: Sequence[float], **kwargs):
        quantiles = ops.cast(quantiles, mixed_precision.dtype_policy().compute_dtype)
        super().__init__(**kwargs, fn=quantile_loss, tau=quantiles)


def quantile_loss(y_true: Array, y_pred: Array, tau: float) -> Array:
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

    y_true = ops.repeat(ops.cast(y_true, y_pred.dtype)[..., newaxis], ops.shape(tau)[0], axis=-1)
    tau = ops.broadcast_to(tau, y_pred.shape)

    prediction_underflow = y_true - y_pred

    over_estimation_error = tau * ops.maximum(prediction_underflow, 0.0)
    under_estimation_error = (1 - tau) * ops.maximum(-prediction_underflow, 0.0)

    # Sum over outputs
    error = ops.sum(over_estimation_error + under_estimation_error, axis=-2)
    # Average over time steps
    quantile_error = ops.mean(error, axis=-2)
    # Sum over quantiles.
    return ops.sum(quantile_error, axis=-1)
