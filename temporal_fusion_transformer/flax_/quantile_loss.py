from __future__ import annotations

from typing import Callable, Sequence
from jaxtyping import Array, Float, Scalar


def make_quantile_loss(
    num_outputs: int, quantiles: Sequence[float] = None
) -> Callable[
    [Float[Array, "batch time_steps n"], Float[Array, "batch time_steps n*q"]],
    Float[Scalar],
]:
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    def quantile_loss(
        y_true: Float[Array, "batch time_steps n"],
        y_pred: Float[Array, "batch time_steps n*q"],
    ) -> Float[Scalar]:
        pass

    return quantile_loss
