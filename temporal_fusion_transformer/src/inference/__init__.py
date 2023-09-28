from __future__ import annotations

from typing import TYPE_CHECKING
import jax.numpy as jnp
from jaxtyping import Array, Float32


if TYPE_CHECKING:
    import tensorflow as tf
    from temporal_fusion_transformer.src.modeling.tft_model import TemporalFusionTransformer
    from temporal_fusion_transformer.src.training.training_lib import ApplyFunc
    from flax.core.frozen_dict import FrozenDict


class InferenceService:
    __slots__ = ["apply_fn", "params"]

    def __init__(self, apply_fn: ApplyFunc, params: FrozenDict):
        self.apply_fn = apply_fn
        self.params = params

    def predict(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        if jnp.ndim(x_batch) == 3:
            return self.apply_fn(self.params, x_batch)
