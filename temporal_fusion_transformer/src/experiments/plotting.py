from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, jaxtyped


@jaxtyped
def plot_predictions(
    x_batch: Float[Array, "batch (t_1+t_2) n"],
    y_batch: Float[Array, "batch (t_1+t_2) m"],
    y_predicted: Float[Array, "batch t_2 m q"],
    quantiles: Float[Array, "q"],
    decode_timestamps_fn: Callable[[Float[Array, "batch*(t_1+t_2) n"]], List[datetime]],
    inverse_transform_inputs_fn: Callable[[Float[Array, "batch*t n"]], Float[Array, "batch*t n"]],
    inverse_transform_targets_fn: Callable[[Float[Array, "batch*t m"]], Float[Array, "batch*t m"]],
    fig_axs_factory: Callable[[], Tuple[plt.Axes, plt.Figure]] | None = None,
) -> plt.Figure:
    num_encoder_steps = y_batch.shape[1] - y_predicted.shape[1]
    num_features = x_batch.shape[-1]
    num_outputs = x_batch.shape[0]

    x_past = x_batch[:, num_encoder_steps:]
    x_future = x_past[:, :num_encoder_steps]
    y_past = y_batch[:, num_encoder_steps:]
    y_future = y_batch[:, :num_encoder_steps]

    if y_future.shape != y_predicted.shape:
        raise RuntimeError("Outputs must have same shape as future values.")

    x_past = np.reshape(x_past, (-1, num_features))
    x_future = np.reshape(x_future, (-1, num_features))

    x_past = inverse_transform_inputs_fn(x_past)
    x_future = inverse_transform_inputs_fn(x_future)

    past_timestamps = decode_timestamps_fn(x_past)
    future_timestamps = decode_timestamps_fn(x_future)

    y_past = np.reshape(y_past, (-1, num_outputs))
    y_future = np.reshape(y_future, (-1, num_outputs))
    # Push quantile axes on first position
    y_predicted = np.transpose(y_predicted, [3, 0, 1, 2])
    y_predicted = [np.reshape(y, (-1, num_outputs)) for y in y_predicted]

    y_past = inverse_transform_targets_fn(y_past)
    y_future = inverse_transform_targets_fn(y_future)
    y_predicted = [inverse_transform_targets_fn(y) for y in y_predicted]

    if fig_axs_factory is None:
        fig, axs = plt.subplots()
    else:
        fig, axs = fig_axs_factory()

    for i in num_outputs:
        axs.plot(
            past_timestamps,
            y_past[..., i],
            label=f"Past Observed Outputs {i}",
            marker=0,
            markersize=2,
        )
        axs.plot(
            future_timestamps,
            y_future,
            label=f"Ground Truth Outputs {i}",
            marker=0,
            markersize=2,
        )

        for y_predicted_i, quantile_i in zip(y_predicted, quantiles):
            axs.plot(
                future_timestamps,
                y_predicted_i,
                marker=0,
                markersize=2,
                label=f"Quantile={quantile_i:.1f} Prediction.",
            )
        axs.legend()  #

        plt.tight_layout()
