from __future__ import annotations

from functools import cached_property
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
from absl import logging
from flax import struct
from jaxtyping import Array, Float, jaxtyped
from ml_collections import ConfigDict

from temporal_fusion_transformer.src.config_dict import ConfigDictProto
from temporal_fusion_transformer.src.tft_layers import (
    ComputeDtype,
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


@struct.dataclass
class TftOutputs:
    logits: Float[Array, "batch time_steps n*quantiles"]
    static_flags: Float[Array, "batch n_s"]
    historical_flags: Float[Array, "batch t (n - n_s)"]
    future_flags: Float[Array, "batch (t -T) (n - n_s)"]


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
    num_decoder_blocks: int = 1
    dropout_rate: float = 0.1
    num_quantiles: int = 3
    # cause by data
    num_outputs: int = 1
    # optional, as sometimes can be deduced from input.
    input_observed_idx: Sequence[int] | None = None
    input_static_idx: Sequence[int] | None = None
    input_known_real_idx: Sequence[int] | None = None
    input_known_categorical_idx: Sequence[int] = None
    num_observed_inputs: int | None = None
    num_known_real_inputs: int | None = None
    num_known_categorical_inputs: int | None = None
    num_static_inputs: int | None = None
    return_attention: bool = False
    dtype: ComputeDtype = jnp.float32

    @nn.compact
    @jaxtyped
    def __call__(
        self, inputs: Float[Array, "batch time n"] | InputStruct, training: bool = False
    ) -> Float[Array, "batch time n*quantiles"] | TftOutputs:
        if not isinstance(inputs, InputStruct):
            inputs = self.make_input_struct(inputs)

        inputs = inputs.cast_inexact(self.dtype)

        embeddings = InputEmbedding(
            static_categories_sizes=self.static_categories_sizes,
            known_categories_sizes=self.known_categories_sizes,
            num_known_real_inputs=self._num_known_real_inputs,
            num_observed_inputs=self._num_observed_inputs,
            latent_dim=self.latent_dim,
            dtype=self.dtype,
        )(inputs)

        static_context = StaticCovariatesEncoder(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_static_inputs=self._num_static_inputs,
            dtype=self.dtype,
        )(embeddings.static, training=training)

        # Isolate known and observed historical inputs.
        historical_inputs = [embeddings.known[:, : self.num_encoder_steps]]
        if embeddings.observed is not None:
            historical_inputs.append(embeddings.observed[:, : self.num_encoder_steps])
        if embeddings.unknown is not None:
            historical_inputs.append(embeddings.unknown[:, : self.num_encoder_steps])

        historical_inputs = jnp.concatenate(historical_inputs, axis=-1)

        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps : self.total_time_steps]
        historical_features, historical_flags, _ = VariableSelectionNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_inputs=self._num_known_real_inputs + self._num_known_categorical_inputs + self._num_observed_inputs,
            num_time_steps=self.num_encoder_steps,
            dtype=self.dtype,
        )(historical_inputs, static_context.enrichment, training=training)

        future_features, future_flags, _ = VariableSelectionNetwork(
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            num_time_steps=self.total_time_steps - self.num_encoder_steps,
            num_inputs=self._num_known_real_inputs + self._num_known_categorical_inputs,
            dtype=self.dtype,
        )(future_inputs, static_context.enrichment, training=training)
        state_carry, history_lstm = nn.RNN(nn.OptimizedLSTMCell(self.latent_dim, dtype=self.dtype), return_carry=True)(
            historical_features,
            initial_carry=(static_context.state_h, static_context.state_c),
        )
        future_lstm = nn.RNN(nn.OptimizedLSTMCell(self.latent_dim, dtype=self.dtype))(
            future_features, initial_carry=state_carry
        )

        lstm_outputs = jnp.concatenate([history_lstm, future_lstm], axis=1)
        input_embeddings = jnp.concatenate([historical_features, future_features], axis=1)

        lstm_outputs, _ = GatedLinearUnit(
            latent_dim=self.latent_dim, dropout_rate=self.dropout_rate, time_distributed=True, dtype=self.dtype
        )(lstm_outputs, training=training)
        temporal_features = nn.LayerNorm(dtype=self.dtype)(lstm_outputs + input_embeddings)

        enriched, _ = GatedResidualNetwork(
            latent_dim=self.latent_dim, dropout_rate=self.dropout_rate, time_distributed=True, dtype=self.dtype
        )(temporal_features, static_context.vector[:, jnp.newaxis], training=training)
        decoder_in = enriched
        mask = make_causal_mask(decoder_in, dtype=self.dtype)
        for _ in range(self.num_decoder_blocks):
            decoder_out = DecoderBlock(
                num_attention_heads=self.num_attention_heads,
                latent_dim=self.latent_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(decoder_in, mask=mask, training=training)
            decoder_out = nn.LayerNorm(dtype=self.dtype)(decoder_out + temporal_features)
            decoder_in = decoder_out

        # Final skip connection
        decoded, _ = GatedLinearUnit(
            latent_dim=self.latent_dim, dropout_rate=self.dropout_rate, time_distributed=True, dtype=self.dtype
        )(decoder_in, training=training)

        outputs = TimeDistributed(
            nn.Dense(self.num_outputs * self.num_quantiles, dtype=self.dtype),
        )(decoder_in[:, self.num_encoder_steps : self.total_time_steps])

        if self.return_attention:
            return TftOutputs(
                logits=outputs,
                historical_flags=historical_flags[..., 0, :],
                future_flags=future_flags[..., 0, :],
                static_flags=static_context.weight[..., 0],
            )
        else:
            return outputs

    @cached_property
    def _num_known_real_inputs(self) -> int:
        if self.num_known_real_inputs is not None:
            return self.num_known_real_inputs
        elif self.input_known_real_idx is not None:
            return len(self.input_known_real_idx)
        else:
            raise ValueError(f"Must provide either `num_known_real_inputs` or input_known_real_idx")

    @cached_property
    def _num_known_categorical_inputs(self) -> int:
        if self.num_known_categorical_inputs is not None:
            return self.num_known_categorical_inputs
        elif self.input_known_categorical_idx is not None:
            return len(self.input_known_categorical_idx)
        else:
            raise ValueError(f"Must provide either `num_known_categorical_inputs` or input_known_categorical_idx")

    @cached_property
    def _num_static_inputs(self) -> int:
        if self.num_static_inputs is not None:
            return self.num_static_inputs
        elif self.input_static_idx is not None:
            return len(self.input_static_idx)
        else:
            raise ValueError(f"Must provide either `num_static_inputs` or input_static_idx")

    @cached_property
    def _num_observed_inputs(self) -> int:
        if self.num_observed_inputs is not None:
            return self.num_observed_inputs
        elif self.input_observed_idx is not None:
            return len(self.input_observed_idx)
        else:
            raise ValueError(f"Must provide either `num_observed_inputs` or input_observed_idx")

    def make_input_struct(self, inputs: jnp.ndarray) -> InputStruct:
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
                unknown_inputs = jnp.take(inputs, unknown_indexes, axis=-1).astype(self.dtype)
            else:
                logging.error(
                    f"Declared number of features does not match with the one seen in input, "
                    f"no inputs at {unknown_indexes}"
                )
                unknown_inputs = None
        else:
            unknown_inputs = None

        static = jnp.take(inputs, jnp.asarray(input_static_idx), axis=-1).astype(jnp.int32)

        if len(input_known_real_idx) > 0:
            known_real = jnp.take(inputs, jnp.asarray(input_known_real_idx), axis=-1).astype(self.dtype)
        else:
            known_real = None

        if len(input_known_categorical_idx) > 0:
            known_categorical = jnp.take(inputs, jnp.asarray(input_known_categorical_idx), axis=-1).astype(jnp.int32)
        else:
            known_categorical = None

        if len(input_observed_idx) > 0:
            observed = jnp.take(inputs, jnp.asarray(input_observed_idx), axis=-1).astype(self.dtype)
        else:
            observed = None

        return InputStruct(
            static=static,
            known_real=known_real,
            known_categorical=known_categorical,
            observed=observed,
            unknown=unknown_inputs,
        )

    @staticmethod
    def from_config_dict(
        config: ConfigDict | ConfigDictProto, jit_module: bool = False, dtype=jnp.float32
    ) -> TemporalFusionTransformer:
        fixed_params = config.fixed_params
        hyperparams = config.hyperparams

        module = TemporalFusionTransformer
        if jit_module:
            module = nn.jit(module, static_argnums=2)

        model = module(
            static_categories_sizes=fixed_params.static_categories_sizes,
            known_categories_sizes=fixed_params.known_categories_sizes,
            latent_dim=hyperparams.latent_dim,
            num_encoder_steps=fixed_params.num_encoder_steps,
            dropout_rate=hyperparams.dropout_rate,
            input_observed_idx=fixed_params.input_observed_idx,
            input_static_idx=fixed_params.input_static_idx,
            input_known_real_idx=fixed_params.input_known_real_idx,
            input_known_categorical_idx=fixed_params.input_known_categorical_idx,
            num_attention_heads=hyperparams.num_attention_heads,
            num_decoder_blocks=hyperparams.num_decoder_blocks,
            num_quantiles=len(hyperparams.quantiles),
            num_outputs=fixed_params.num_outputs,
            total_time_steps=fixed_params.total_time_steps,
            dtype=dtype,
        )
        return model
