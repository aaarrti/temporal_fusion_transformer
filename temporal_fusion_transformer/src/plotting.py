from __future__ import annotations

from typing import Sequence, Protocol, runtime_checkable, TypeVar, TYPE_CHECKING
import numpy as np
import tensorflow as tf

T = TypeVar("T")
K = TypeVar("K")

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@runtime_checkable
class SupportGetItem(Protocol[T]):
    def __getitem__(self, item: int) -> T:
        ...


def plot_predictions(
    predicted_outputs: np.ndarray,
    future_timestamps: np.ndarray,
    num_outputs: int,  # n
    quantiles: np.ndarray,
    future_outputs: np.ndarray,
    past_time_stamps: np.ndarray,
    past_outputs: np.ndarray,
    output_labels: Sequence[str] | None = None,
    target_scaler=None,
) -> plt.Figure:
    """

    Parameters
    ----------
    predicted_outputs:
        models outputs, with shape (batch, timestamps, len(quantiles) * num_outputs)
    future_timestamps:
        time stamps
    num_outputs:
        Number of entities to be predicted.
    future_outputs:
        Ground truth values to compare predictions with.
    past_time_stamps:
    past_outputs:
    quantiles
    output_labels:
    target_scaler:

    Returns
    -------

    """
    import matplotlib.pyplot as plt

    tf.debugging.assert_rank(predicted_outputs, 3)
    tf.debugging.assert_rank(future_timestamps, 2)
    tf.debugging.assert_rank(past_time_stamps, 2)
    tf.debugging.assert_rank(past_outputs, 3)
    tf.debugging.assert_rank(future_outputs, 3)

    def _noop_scaler(_, arr: np.ndarray) -> np.ndarray:
        return arr

    if target_scaler is None:
        target_scaler = _noop_scaler

    if output_labels is None:
        output_labels = [f"Output {i+1}" for i in range(num_outputs)]

    future_sort_idx = np.argsort(np.reshape(future_timestamps, -1))
    future_timestamps = np.take(np.reshape(future_timestamps, -1), future_sort_idx)
    prediction_shape = np.shape(predicted_outputs)
    batch_size, n_time_steps = prediction_shape[0], prediction_shape[1]
    predicted_outputs = np.reshape(
        predicted_outputs, (batch_size, n_time_steps, num_outputs, len(quantiles))
    )

    past_sort_idx = np.argsort(np.reshape(past_time_stamps, -1))
    past_time_stamps = np.take(np.reshape(past_time_stamps, -1), past_sort_idx)

    fig: plt.Figure
    fig, axs = plt.subplots(num_outputs, 1, sharex="row")

    axs = wrap_axes(axs)
    for i, label in enumerate(output_labels):
        past_output_i = target_scaler(label, past_outputs[..., i])
        future_output_i = target_scaler(label, future_outputs[..., i])

        axs[i].set_title(label)
        axs[i].set(ylabel="Time, ???", xlabel="???")

        past_output_i = np.take(np.reshape(past_output_i, -1), past_sort_idx)
        axs[i].plot(
            past_time_stamps,
            past_output_i,
            label="Past Observed Outputs",
            marker=0,
            markersize=2,
        )

        future_output_i = np.take(np.reshape(future_output_i, -1), future_sort_idx)
        axs[i].plot(
            future_timestamps,
            future_output_i,
            label="Ground Truth Outputs",
            marker=0,
            markersize=2,
        )

        for q_i, quantile in enumerate(quantiles):
            qi_prediction = predicted_outputs[:, :, i, q_i]
            qi_prediction = target_scaler(label, qi_prediction)
            qi_prediction = np.take(np.reshape(qi_prediction, -1), future_sort_idx)
            axs[i].plot(
                future_timestamps,
                qi_prediction,
                marker=0,
                markersize=2,
                label=f"Quantile={quantile:.1f} Prediction.",
            )
        axs[i].legend()

    plt.tight_layout()
    return fig


def wrap_axes(axs: plt.Axes | SupportGetItem[plt.Axes]) -> SupportGetItem[plt.Axes]:
    if not isinstance(axs, SupportGetItem):
        return [axs]
    else:
        return axs
