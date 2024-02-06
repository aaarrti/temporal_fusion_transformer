from __future__ import annotations

from collections.abc import Sequence

from jax import Array
from keras import mixed_precision, ops
from keras.src.losses import LossFunctionWrapper, mean_absolute_percentage_error
from keras.src.metrics import MeanMetricWrapper

from temporal_fusion_transformer.src.ops import named_scope


class QuantilePinballLoss(LossFunctionWrapper):
    def __init__(self, quantiles: Sequence[float], name: str = "qp_loss", **kwargs):
        quantiles = ops.cast(quantiles, mixed_precision.dtype_policy().compute_dtype)
        super().__init__(name=name, **kwargs, fn=quantile_pinball_loss, tau=quantiles)


class QuantileMAPE(MeanMetricWrapper):
    def __init__(self, quantiles: Sequence[float], name: str = "q_mape", **kwargs):
        quantiles = ops.cast(quantiles, mixed_precision.dtype_policy().compute_dtype)
        super().__init__(name=name, **kwargs, fn=quantile_mape, tau=quantiles)


def quantile_mape(y_true: Array, y_pred: Array, tau: Array) -> Array:
    with named_scope("q_mape"):
        tau = ops.cast(tau, y_pred.dtype)
        y_true = ops.cast(y_true, y_pred.dtype)
        y_pred = ops.transpose(y_pred, [3, 0, 1, 2])

        err = ops.vectorized_map(lambda x: mean_absolute_percentage_error(y_true, x), y_pred)

        return ops.mean(
            # weighted mean over quantiles
            ops.mean(ops.transpose(err, [1, 2, 0]) * tau, axis=-1),
            # harmonic means over timestamps (same as datapoints)
            axis=-1,
        )


def quantile_pinball_loss(y_true: Array, y_pred: Array, tau: Array) -> Array:
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
    return ops.mean(q_loss, axis=(-1, -2, -3))
