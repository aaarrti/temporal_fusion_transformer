from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import RRuleLocator, ticker

DataTuple = Tuple[List[datetime], np.ndarray]


def plot_predictions(
    ground_truth_past: DataTuple,
    ground_truth_observed: DataTuple,
    predictions: List[DataTuple],
    quantiles: List[float] | None = None,
    title: str = None,
    formatter: ticker.Formatter | None = None,
    locator: RRuleLocator | None = None,
    figsize=(7, 4),
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)

    # TODO: support multiple outputs
    plt.plot(
        ground_truth_past[0],
        ground_truth_past[1],
        marker=0,
        markersize=2,
    )
    plt.plot(
        ground_truth_observed[0],
        ground_truth_observed[1],
        marker=0,
        markersize=2,
    )

    if quantiles is None:
        quantiles = range(len(predictions))

    legend = []
    for i, j in zip(predictions, quantiles):
        label = f"Quantile={j:.1f}" if quantiles is not None else f"Quantile_{j}"
        legend.append(label)
        plt.plot(
            i[0],
            i[1],
            marker=0,
            markersize=2,
        )

    plt.xticks(rotation=90, fontsize=12)

    if title is not None:
        plt.title(title)
    if formatter is not None:
        plt.gca().xaxis.set_major_formatter(formatter)

    if locator is not None:
        plt.gca().xaxis.set_major_locator(locator)

    plt.legend(["Past", "Ground truth"] + legend)
    plt.tight_layout()
    return fig
