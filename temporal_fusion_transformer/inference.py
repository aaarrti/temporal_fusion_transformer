from __future__ import annotations

from typing import Sequence
import tensorflow as tf


def extract_predictions(logits: tf.Tensor, quantiles: Sequence[float]):
    # TODO:
    # - apply inverse transform (need to persis scalers?)
    # - group by identifier and time (need to preserve them from input dataset.)
    # extract prediction for each quantile separately.
    # map predictions back into input feature domain???
    pass
