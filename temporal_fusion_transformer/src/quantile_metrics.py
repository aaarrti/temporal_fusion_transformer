from __future__ import annotations

from collections.abc import Sequence

from keras_core import metrics

from temporal_fusion_transformer.src.utils import enumerate_v2


class QuantileRMSE(metrics.RootMeanSquaredError):
    def __init__(
        self, tau: float, quantile_index: int, output_index: int | None = None, dtype=None
    ):
        name = f"rmse_q({tau:.1f})"
        if output_index is not None:
            name = f"{name}_#{output_index}"
        else:
            output_index = 0

        super().__init__(name, dtype=dtype)
        self.output_index = output_index
        self.quantile_index = quantile_index
        self.tau = tau

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[..., self.output_index, self.quantile_index]
        y_true = y_true[..., self.output_index]
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.tau * super().result()


def make_quantile_rmse_metrics(
    quantiles: Sequence[float],
    num_outputs: int = 1,
) -> list[QuantileRMSE]:
    metrics = []

    if num_outputs == 1:
        # Only one output -> no need to append indexes
        for q_i, q in enumerate_v2(quantiles):
            metrics.append(QuantileRMSE(tau=q, quantile_index=q_i))
        return metrics

    for i in range(num_outputs):
        for q_i, q in enumerate_v2(quantiles):
            metrics.append(QuantileRMSE(tau=q, quantile_index=q_i, output_index=i))

    return metrics
