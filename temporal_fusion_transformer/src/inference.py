from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import FeatureSpace

PredictFn = Callable


"""
We receive as input a grid like data [id, time, values * quantiles]
We also have a grid like structure of [id, time] of actual identifier
We could map those nodes 1 to 1, to make outputs resonable.
"""


class InferenceService:
    def __init__(
        self,
        model: PredictFn,
        feature_space: FeatureSpace,
        batch_size: int | None = None,
    ):
        self.model = model
        self.target_scaler = target_scaler
        self.batch_size = batch_size

    def plot_predictions_for_entity(
        self,
        test_ds: Mapping[str, np.ndarray],
        entity_id: str,
        output_index: int = 0,
        output_name: str = "Output #0",
        fig_axs_factory: Callable[[], Tuple[plt.Figure, plt.Axes]] | None = None,
    ):
        """
        it is up to user to ensure dataset has data only for 1 entity.

        {
        'identifier': (64, 1),
        'time': (64, 192, 1),
        'outputs': (64, 192, 1),
        'inputs_static': (64, 1),
        'inputs_known_real': (64, 192, 3)
        }
        """

        for k in (
            "identifier",
            "time",
            "inputs_static",
            "inputs_known_real",
        ):
            if k not in test_ds:
                raise ValueError(f"Dataset does not contain key {k}")

        x_batch = map_dict(
            filter_dict(test_ds, key_filter=lambda k: "inputs_" in k),
            key_mapper=lambda k: k.replace("inputs_", ""),
        )

        if self.batch_size is not None:
            batch_size = self.batch_size
        else:
            batch_size = len(x_batch["static"])

        logits = self.model.predict(
            x_batch,
            batch_size=batch_size,
        )

        time = test_ds["time"]

        num_past_steps = self.model.num_encoder_steps
        num_steps = time.shape[1]
        num_future_steps = num_steps - num_past_steps

        if logits.shape[1] != num_future_steps:
            raise ValueError(f"Expected model to return {num_future_steps} timestamps, but received {logits.shape[1]}")

        past_time = time[:, :num_past_steps, 0]
        future_time = time[:, num_past_steps:, 0]

        y_batch = test_ds["outputs"]
        past_outputs = y_batch[:, :num_past_steps, output_index]
        future_outputs = y_batch[:, num_past_steps:, output_index]

        past_outputs = self.target_scaler(entity_id, past_outputs)
        future_outputs = self.target_scaler(entity_id, future_outputs)

        future_sort_indexes = np.argsort(np.reshape(future_time, -1))
        past_sort_indexes = np.argsort(np.reshape(past_time, -1))

        future_time = np.take(np.reshape(future_time, -1), future_sort_indexes)
        past_time = np.take(np.reshape(past_time, -1), past_sort_indexes)

        past_outputs = np.take(np.reshape(past_outputs, -1), past_sort_indexes)
        future_outputs = np.take(np.reshape(future_outputs, -1), future_sort_indexes)

        batch_size = logits.shape[0]

        logits = np.reshape(logits, (batch_size, num_future_steps, -1, len(self.model.quantiles)))

        if fig_axs_factory is None:
            fig, axs = plt.subplots()
        else:
            fig, axs = fig_axs_factory()

        axs.set_title(entity_id)
        axs.set(ylabel=output_name, xlabel="Time")
        axs.plot(
            past_time,
            past_outputs,
            label="Past Observed Outputs",
            marker=0,
            markersize=2,
        )
        axs.plot(
            future_time,
            future_outputs,
            label="Ground Truth Outputs",
            marker=0,
            markersize=2,
        )
        axs.set_xticklabels(axs.get_xticks(), rotation=90)

        for q_i, quantile in enumerate(self.model.quantiles):
            qi_prediction = logits[..., output_index, q_i]
            qi_prediction = self.target_scaler(entity_id, qi_prediction)

            qi_prediction = np.take(np.reshape(qi_prediction, -1), future_sort_indexes)
            axs.plot(
                future_time,
                qi_prediction,
                marker=0,
                markersize=2,
                label=f"Quantile={quantile:.1f} Prediction.",
            )
        axs.legend()

        plt.tight_layout()
        return fig
