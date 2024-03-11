from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, TYPE_CHECKING

from temporal_fusion_transformer.modeling.layers import (
    EmbeddingStruct,
    StaticCovariatesEncoder,
    VariableSelectionNetwork,
    GatedLinearUnit,
    GatedResidualNetwork,
    DecoderBlock,
    TimeDistributed,
)

if TYPE_CHECKING:
    from temporal_fusion_transformer.modeling.layers import ComputeDtype


@struct.dataclass
class TftOutputs:
    logits: jax.Array
    static_flags: jax.Array
    historical_flags: jax.Array
    future_flags: jax.Array


class TemporalFusionTransformer(nn.Module):
    """

    Attributes
    ----------

    num_encoder_steps:
        Number of time steps, which will be considered as past.
    dropout_rate:
        Dropout rate passed down to keras.layer.Dropout.
    latent_dim:
        Latent space dimensionality.
    num_attention_heads:
        Number of attention heads to use for multi-head attention.
    num_outputs:
        Number of values to predict.
    num_quantiles:
        Number of quantiles, used to divide predicted distribution, typically you would use 3.
    num_decoder_blocks:
        Number of decoder blocks to apply sequentially.
    total_time_steps:
        Size of the 3rd axis in the input.
    embedding_layer:

    num_non_static_inputs:

    num_known_inputs:

    num_static_inputs:


    References
    ----------

    Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/pdf/1912.09363.pdf, https://github.com/google-research/google-research/tree/master/tft

    """

    # caused by data
    num_encoder_steps: int
    total_time_steps: int
    # hyperparameters
    latent_dim: int
    num_attention_heads: int
    embedding_layer: Callable[[jax.Array], EmbeddingStruct]
    num_non_static_inputs: int
    num_known_inputs: int
    num_static_inputs: int
    num_decoder_blocks: int = 1
    dropout_rate: float = 0.1
    num_quantiles: int = 3
    # cause by data
    num_outputs: int = 1
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array, training: bool = False) -> TftOutputs:
        """

        Parameters
        ----------
        inputs
        training

        Returns
        -------

        """
        embeddings = self.embedding_layer(inputs)

        static_context = StaticCovariatesEncoder(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_static_inputs=self.num_static_inputs,
            dtype=self.dtype,
        )(embeddings.static, training=training)

        # Isolate known and observed historical inputs.
        historical_inputs = [embeddings.known[:, : self.num_encoder_steps]]
        if embeddings.observed is not None:
            historical_inputs.append(embeddings.observed[:, : self.num_encoder_steps])
        historical_inputs = jnp.concatenate(historical_inputs, axis=-1)

        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps : self.total_time_steps]
        historical_features, historical_flags, _ = VariableSelectionNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_inputs=self.num_non_static_inputs,
            num_time_steps=self.num_encoder_steps,
            dtype=self.dtype,
        )(historical_inputs, static_context.enrichment, training=training)

        future_features, future_flags, _ = VariableSelectionNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_time_steps=self.total_time_steps - self.num_encoder_steps,
            num_inputs=self.num_known_inputs,
            dtype=self.dtype,
        )(future_inputs, static_context.enrichment, training=training)
        state_carry, history_lstm = nn.RNN(
            nn.OptimizedLSTMCell(self.latent_dim, dtype=self.dtype),
            return_carry=True,
            # unroll=self.lstm_unroll,
            # split_rngs={"param": self.lstm_split_rngs}
        )(
            historical_features,
            initial_carry=(static_context.state_h, static_context.state_c),
        )
        future_lstm = nn.RNN(
            nn.OptimizedLSTMCell(self.latent_dim, dtype=self.dtype),
            # unroll=self.lstm_unroll,
            # split_rngs={"param": self.lstm_split_rngs}
        )(
            future_features,
            initial_carry=state_carry,
        )

        lstm_outputs = jnp.concatenate([history_lstm, future_lstm], axis=1)
        input_embeddings = jnp.concatenate([historical_features, future_features], axis=1)

        lstm_outputs, _ = GatedLinearUnit(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
        )(lstm_outputs, training=training)
        temporal_features = nn.LayerNorm(dtype=self.dtype)(lstm_outputs + input_embeddings)

        enriched, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
        )(temporal_features, static_context.vector[:, jnp.newaxis], training=training)
        decoder_in = enriched

        for _ in range(self.num_decoder_blocks):
            decoder_out = DecoderBlock(
                num_attention_heads=self.num_attention_heads,
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(decoder_in, training=training)
            decoder_out = nn.LayerNorm(dtype=self.dtype)(decoder_out + temporal_features)
            decoder_in = decoder_out

        outputs = TimeDistributed(
            nn.Dense(self.num_outputs * self.num_quantiles, dtype=self.dtype),
        )(decoder_in[:, self.num_encoder_steps : self.total_time_steps])
        outputs = jnp.reshape(
            outputs,
            (
                -1,
                self.total_time_steps - self.num_encoder_steps,
                self.num_outputs,
                self.num_quantiles,
            ),
        )
        return TftOutputs(
            logits=outputs,
            historical_flags=historical_flags[..., 0, :],
            future_flags=future_flags[..., 0, :],
            static_flags=static_context.weight[..., 0],
        )
