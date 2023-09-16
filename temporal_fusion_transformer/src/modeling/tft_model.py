from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import flax.linen as nn
import jax.numpy as jnp
from absl import logging
from flax import struct
from jaxtyping import Array, Float, jaxtyped

from temporal_fusion_transformer.src.modeling.tft_layers import (
    DecoderBlock,
    GatedLinearUnit,
    GatedResidualNetwork,
    InputEmbedding,
    InputStruct,
    StaticCovariatesEncoder,
    TimeDistributed,
    VariableSelectionNetwork,
    make_causal_mask,
)

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.config_dict import ConfigDict, DatasetConfig
    from temporal_fusion_transformer.src.modeling.tft_layers import ComputeDtype


@struct.dataclass
class TftOutputs:
    logits: Float[Array, "batch time_steps n q"]
    static_flags: Float[Array, "batch n_s"]
    historical_flags: Float[Array, "batch t (n-n_s)"]
    future_flags: Float[Array, "batch (t-T) (n-n_s)"]


class TemporalFusionTransformer(nn.Module):
    """
    References
    ----------

    Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/pdf/1912.09363.pdf, https://github.com/google-research/google-research/tree/master/tft

    """

    input_preprocessor: InputPreprocessor
    input_embedding: InputEmbedding
    static_covariates_encoder: StaticCovariatesEncoder
    historical_variable_selection: VariableSelectionNetwork
    future_variable_selection: VariableSelectionNetwork
    historical_rnn: nn.RNN
    future_rnn: nn.RNN
    lstm_skip_connection: GatedLinearUnit
    static_context_skip_connection: GatedLinearUnit
    output_skip_connection: GatedLinearUnit
    decoder_blocks: List[DecoderBlock]
    output_projection: List[nn.Module]

    num_encoder_steps: int
    total_time_steps: int

    # ...
    return_attention: bool = False
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    @jaxtyped
    def __call__(
        self, inputs: Float[Array, "batch time n"], training: bool = False
    ) -> Float[Array, "batch time n q"] | TftOutputs:
        inputs = self.input_preprocessor(inputs)
        embeddings = self.input_embedding(inputs)
        static_context = self.static_covariates_encoder(embeddings.static, training=training)

        # Isolate known and observed historical inputs.
        historical_inputs = [embeddings.known[:, : self.num_encoder_steps]]
        if embeddings.observed is not None:
            historical_inputs.append(embeddings.observed[:, : self.num_encoder_steps])
        if embeddings.unknown is not None:
            historical_inputs.append(embeddings.unknown[:, : self.num_encoder_steps])

        historical_inputs = jnp.concatenate(historical_inputs, axis=-1)

        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps : self.total_time_steps]
        historical_features, historical_flags, _ = self.historical_variable_selection(
            historical_inputs, static_context.enrichment, training=training
        )
        future_features, future_flags, _ = self.future_variable_selection(
            future_inputs, static_context.enrichment, training=training
        )

        state_carry, history_lstm = self.historical_rnn(
            historical_features,
            initial_carry=(static_context.state_h, static_context.state_c),
        )
        future_lstm = self.future_rnn(future_features, initial_carry=state_carry)

        lstm_outputs = jnp.concatenate([history_lstm, future_lstm], axis=1)
        input_embeddings = jnp.concatenate([historical_features, future_features], axis=1)

        lstm_outputs, _ = self.lstm_skip_connection(lstm_outputs, training=training)
        temporal_features = nn.LayerNorm(dtype=self.dtype)(lstm_outputs + input_embeddings)

        enriched, _ = self.static_context_skip_connection(
            temporal_features, static_context.vector[:, jnp.newaxis], training=training
        )
        decoder_in = enriched
        mask = make_causal_mask(decoder_in, dtype=self.dtype)

        for block in self.decoder_blocks:
            decoder_out = block(decoder_in, mask=mask, training=training)
            decoder_out = nn.LayerNorm(dtype=self.dtype)(decoder_out + temporal_features)
            decoder_in = decoder_out

        # Final skip connection
        decoded, _ = self.output_skip_connection(decoder_in, training=training)

        outputs = []

        for layer in self.output_projection:
            outputs_i = layer(decoded[:, self.num_encoder_steps : self.total_time_steps])
            outputs.append(outputs_i)

        outputs = jnp.stack(outputs, axis=-1)

        if self.return_attention:
            return TftOutputs(
                logits=outputs,
                historical_flags=historical_flags[..., 0, :],
                future_flags=future_flags[..., 0, :],
                static_flags=static_context.weight[..., 0],
            )
        else:
            return outputs

    @staticmethod
    def from_config_dict(
        config: ConfigDict,
        data_config: DatasetConfig,
        jit_module: bool = False,
        dtype: jnp.inexact = jnp.float32,
    ) -> TemporalFusionTransformer:
        return make_temporal_fusion_transformer(config, data_config, jit_module, dtype)


class InputPreprocessor(nn.Module):
    input_observed_idx: Sequence[int]
    input_static_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]
    dtype: jnp.inexact = jnp.float32

    @nn.compact
    def __call__(self, inputs: Float[Array, "batch time n"]) -> InputStruct:
        input_static_idx, input_known_real_idx, input_known_categorical_idx, input_observed_idx = (
            self.input_static_idx,
            self.input_known_real_idx,
            self.input_known_categorical_idx,
            self.input_observed_idx,
        )

        if input_static_idx is None:
            raise ValueError(f"When providing inputs as arrays, must specify provide `input_static_idx`")

        if input_known_real_idx is None:
            raise ValueError(f"When providing inputs as arrays, must specify provide `input_known_real_idx`")

        if input_known_categorical_idx is None:
            raise ValueError(f"When providing inputs as arrays, must specify provide `input_known_categorical_idx`")

        if input_observed_idx is None:
            raise ValueError(f"When providing inputs as arrays, must specify provide `input_observed_idx`")

        input_static_idx = list(input_static_idx)
        input_known_real_idx = list(input_known_real_idx)
        input_known_categorical_idx = list(input_known_categorical_idx)
        input_observed_idx = list(input_observed_idx)

        declared_num_features = (
            len(input_static_idx)
            + len(input_known_real_idx)
            + len(input_known_categorical_idx)
            + len(input_observed_idx)
        )
        num_features = inputs.shape[-1]

        if num_features != declared_num_features:
            unknown_indexes = sorted(
                list(
                    set(
                        input_static_idx + input_known_real_idx + input_known_categorical_idx + input_observed_idx
                    ).symmetric_difference(range(num_features))
                )
            )
            if num_features > declared_num_features:
                logging.error(
                    f"Declared number of features does not match with the one seen in input, "
                    f"could not indentify inputs at {unknown_indexes}"
                )
                unknown_indexes = jnp.asarray(unknown_indexes, jnp.int32)
                unknown_inputs = inputs[..., unknown_indexes].astype(self.dtype)
            else:
                logging.error(
                    f"Declared number of features does not match with the one seen in input, "
                    f"no inputs at {unknown_indexes}"
                )
                unknown_inputs = None
        else:
            unknown_inputs = None

        static = inputs[..., input_static_idx].astype(jnp.int32)

        if len(input_known_real_idx) > 0:
            known_real = inputs[..., input_known_real_idx].astype(self.dtype)
        else:
            known_real = None

        if len(input_known_categorical_idx) > 0:
            known_categorical = inputs[..., input_known_categorical_idx].astype(jnp.int32)
        else:
            known_categorical = None

        if len(input_observed_idx) > 0:
            observed = inputs[..., input_observed_idx].astype(self.dtype)
        else:
            observed = None

        return InputStruct(
            static=static,
            known_real=known_real,
            known_categorical=known_categorical,
            observed=observed,
            unknown=unknown_inputs,
        )


def make_temporal_fusion_transformer(
    config: ConfigDict,
    data_config: DatasetConfig,
    jit_module: bool = False,
    dtype: jnp.inexact = jnp.float32,
) -> TemporalFusionTransformer:
    module = TemporalFusionTransformer
    if jit_module:
        module = nn.jit(module, static_argnums=2)

    num_known_real_inputs = len(data_config.input_known_real_idx)
    num_known_categorical_inputs = len(data_config.input_known_categorical_idx)
    num_observed_inputs = len(data_config.input_observed_idx)
    latent_dim = config.model.latent_dim
    dropout_rate = config.model.dropout_rate
    num_encoder_steps = data_config.num_encoder_steps
    total_time_steps = data_config.total_time_steps

    input_preprocessor = InputPreprocessor(
        input_observed_idx=data_config.input_observed_idx,
        input_static_idx=data_config.input_static_idx,
        input_known_real_idx=data_config.input_known_real_idx,
        input_known_categorical_idx=data_config.input_known_categorical_idx,
        dtype=dtype,
    )
    input_embedding = InputEmbedding(
        static_categories_sizes=data_config.static_categories_sizes,
        known_categories_sizes=data_config.known_categories_sizes,
        num_known_real_inputs=num_known_real_inputs,
        num_observed_inputs=num_observed_inputs,
        latent_dim=latent_dim,
        dtype=dtype,
    )

    static_covariates_encoder = StaticCovariatesEncoder(
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        num_static_inputs=len(data_config.static_categories_sizes),
        dtype=dtype,
    )

    historical_variable_selection = VariableSelectionNetwork(
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        num_inputs=num_known_real_inputs + num_known_categorical_inputs + num_observed_inputs,
        num_time_steps=num_encoder_steps,
        dtype=dtype,
    )

    future_variable_selection = VariableSelectionNetwork(
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        num_time_steps=total_time_steps - num_encoder_steps,
        num_inputs=num_known_real_inputs + num_known_categorical_inputs,
        dtype=dtype,
    )
    historical_rnn = nn.RNN(
        nn.OptimizedLSTMCell(latent_dim, dtype=dtype), return_carry=True, split_rngs={"params": False, "lstm": True}
    )

    future_rnn = nn.RNN(nn.OptimizedLSTMCell(latent_dim, dtype=dtype), split_rngs={"params": False, "lstm": True})

    lstm_skip_connection = GatedLinearUnit(
        latent_dim=latent_dim, dropout_rate=dropout_rate, time_distributed=True, dtype=dtype
    )

    static_context_skip_connection = GatedResidualNetwork(
        latent_dim=latent_dim, dropout_rate=dropout_rate, time_distributed=True, dtype=dtype
    )

    output_skip_connection = GatedLinearUnit(
        latent_dim=latent_dim, dropout_rate=dropout_rate, time_distributed=True, dtype=dtype
    )

    decoder_blocks = [
        DecoderBlock(
            num_attention_heads=config.model.num_attention_heads,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )
        for _ in range(config.model.num_decoder_blocks)
    ]

    output_projection = [
        TimeDistributed(nn.Dense(data_config.num_outputs, dtype=dtype)) for _ in range(len(config.model.quantiles))
    ]

    return module(
        input_preprocessor=input_preprocessor,
        input_embedding=input_embedding,
        static_context_skip_connection=static_context_skip_connection,
        static_covariates_encoder=static_covariates_encoder,
        decoder_blocks=decoder_blocks,
        output_projection=output_projection,
        output_skip_connection=output_skip_connection,
        lstm_skip_connection=lstm_skip_connection,
        future_rnn=future_rnn,
        historical_rnn=historical_rnn,
        future_variable_selection=future_variable_selection,
        historical_variable_selection=historical_variable_selection,
        num_encoder_steps=num_encoder_steps,
        total_time_steps=total_time_steps,
        dtype=dtype,
    )
