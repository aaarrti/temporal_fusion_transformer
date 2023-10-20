from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from keras_core import ops
from keras_core.mixed_precision import global_policy
from keras_core.src.metrics.reduction_metrics import MeanMetricWrapper

from temporal_fusion_transformer.src.utils import enumerate_v2

if TYPE_CHECKING:
    from jax import Array


class OverEstimationError(MeanMetricWrapper):
    def __init__(self, tau: float, quantile_index: int, output_index: int = 0):
        name = f"over_estimation_error_{tau:.2f}_{output_index}"
        super().__init__(
            fn=over_estimation_error,
            dtype=global_policy().compute_dtype,
            name=name,
            tau=tau,
            q_index=quantile_index,
            output_index=output_index,
        )


class UnderEstimationError(MeanMetricWrapper):
    def __init__(self, tau: float, quantile_index: int, output_index: int = 0):
        name = f"under_estimation_error_{tau:.2f}_{output_index}"
        super().__init__(
            fn=under_estimation_error,
            dtype=global_policy().compute_dtype,
            name=name,
            tau=tau,
            q_index=quantile_index,
            output_index=output_index,
        )


def _compute_underflow(y_true: Array, y_pred: Array, q_index: int, output_index: int) -> Array:
    y_pred = y_pred[..., q_index][..., output_index]
    y_true = ops.cast(y_true[..., output_index], y_pred.dtype)
    return y_true - y_pred


def over_estimation_error(y_true: Array, y_pred: Array, tau: float, q_index: int, output_index: int) -> Array:
    tau = ops.cast(tau, y_pred.dtype)
    prediction_underflow = _compute_underflow(y_true, y_pred, q_index, output_index)
    error = tau * ops.maximum(prediction_underflow, 0.0)
    # Average over timestamps
    return ops.mean(error, axis=1)


def under_estimation_error(y_true: Array, y_pred: Array, tau: float, q_index: int, output_index: int) -> Array:
    tau = ops.cast(tau, y_pred.dtype)
    prediction_underflow = _compute_underflow(y_true, y_pred, q_index, output_index)
    error = (1 - tau) * ops.maximum(-prediction_underflow, 0.0)
    # Average over timestamps
    return ops.mean(error, axis=1)


def make_quantile_error_metrics(
    quantiles: Sequence[float],
    num_outputs: int = 1,
) -> List[OverEstimationError | UnderEstimationError]:
    metrics = []

    for i in range(num_outputs):
        for q_i, q in enumerate_v2(quantiles):
            metrics.append(OverEstimationError(tau=q, quantile_index=q_i, output_index=i))

            metrics.append(UnderEstimationError(tau=q, quantile_index=q_i, output_index=i))

    return metrics
