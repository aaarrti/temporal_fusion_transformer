from __future__ import annotations

from typing import Mapping, Tuple, List, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax


class TemporalFusionTransformer(nn.Module):
    pass


class TFTInputEmbedding(nn.Module):
    static_categories_size: List[int]
    known_categories_size: List[int]
    hidden_layer_size: int

    @nn.compact
    def __call__(
        self, inputs: Mapping[str, jnp.ndarray], **kwargs
    ) -> Dict[str, jnp.ndarray]:
        static = inputs["static"]
        known_real = inputs["known_real"]
        known_categorical = inputs.get("known_categorical")
        observed = inputs.get("observed")

        static_input_embeddings = []
        known_real_inputs_embeddings = []


class StaticCovariatesEncoder(nn.Module):
    pass


class TemporalVariableSelectionNetwork(nn.Module):
    pass


class TemporalFusionDecoder(nn.Module):
    pass


class TimeDistributed(nn.Module):
    pass


class GatedLinearUnit(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool
    name: str = "glu"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Args:
            inputs:
        """
        x = nn.Dropout(rate=self.dropout_rate)(inputs)
        if self.use_time_distributed:
            x_pre_activation = TimeDistributed(nn.Dense(self.hidden_layer_size))(x)
            x_gated = TimeDistributed(nn.sigmoid)(x)
        else:
            x_pre_activation = nn.Dense(self.hidden_layer_size)(x)
            x_gated = nn.sigmoid(x)

        x = x_pre_activation * x_gated
        return x


class GatedResidualNetwork(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool
    output_size: int | None = None

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray | Mapping[str, jnp.ndarray], **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass


class TemporalEncoderBlock(nn.Module):
    pass


def make_causal_attention_mask(self_attn_inputs):
    pass
