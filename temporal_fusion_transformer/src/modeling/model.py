from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from temporal_fusion_transformer.src.modeling.layers import (
    InputEmbedding,
    ComputeDtype,
    StaticCovariatesEncoder,
    VariableSelectionNetwork,
    GatedLinearUnit,
    GatedResidualNetwork,
    DecoderBlock,
    TimeDistributed,
)


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

    static_categories_sizes:
        List with maximum value for each category of static inputs in order.
    known_categories_sizes:
        List with maximum value for each category of known categorical inputs in order.
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
    input_observed_idx:
        Indices in 3rd axis in input tensor, which have observed inputs.
    input_static_idx:
        Indices in 3rd axis in input tensor, which have static inputs.
    input_known_real_idx:
        Indices in 3rd axis in input tensor, which have real-valued known inputs.
    input_known_categorical_idx:
        Indices in 3rd axis in input tensor, which have categorical known inputs.
    num_decoder_blocks:
        Number of decoder blocks to apply sequentially.
    total_time_steps:
        Size of the 3rd axis in the input.

    References
    ----------

    Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/pdf/1912.09363.pdf, https://github.com/google-research/google-research/tree/master/tft

    """

    # caused by data
    static_categories_sizes: Sequence[int]
    known_categories_sizes: Sequence[int]
    num_encoder_steps: int
    total_time_steps: int
    # hyperparameters
    latent_dim: int
    num_attention_heads: int
    input_observed_idx: Sequence[int]
    input_static_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]
    num_decoder_blocks: int = 1
    dropout_rate: float = 0.1
    num_quantiles: int = 3
    # cause by data
    num_outputs: int = 1
    dtype: ComputeDtype = jnp.float32
    lstm_unroll: int = 1
    lstm_split_rngs: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.static_categories_sizes) != len(self.input_static_idx):
            raise ValueError("len(self.static_categories_sizes) != len(self.input_static_idx)")

        if len(self.known_categories_sizes) != len(self.input_known_categorical_idx):
            raise ValueError(
                "len(self.known_categories_sizes) != len(self.input_known_categorical_idx)"
            )

        if len(self.input_static_idx) == 0:
            raise ValueError("Must provider at least one static input, e.g., id")

    @nn.compact
    def __call__(self, inputs: jax.Array, training: bool = False) -> jax.Array:
        """

        Parameters
        ----------
        inputs
        training

        Returns
        -------

        """
        embeddings = InputEmbedding(
            static_categories_sizes=self.static_categories_sizes,
            known_categories_sizes=self.known_categories_sizes,
            input_static_idx=self.input_static_idx,
            input_known_real_idx=self.input_known_real_idx,
            input_known_categorical_idx=self.input_known_categorical_idx,
            input_observed_idx=self.input_observed_idx,
            latent_dim=self.latent_dim,
            dtype=self.dtype,
        )(inputs)

        static_context = StaticCovariatesEncoder(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_static_inputs=len(self.input_static_idx),
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
            num_inputs=len(self.input_known_real_idx)
            + len(self.input_known_categorical_idx)
            + len(self.input_observed_idx),
            num_time_steps=self.num_encoder_steps,
            dtype=self.dtype,
        )(historical_inputs, static_context.enrichment, training=training)

        future_features, future_flags, _ = VariableSelectionNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_time_steps=self.total_time_steps - self.num_encoder_steps,
            num_inputs=len(self.input_known_real_idx) + len(self.input_known_categorical_idx),
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
