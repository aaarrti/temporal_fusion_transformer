from __future__ import annotations

from typing import Tuple, List

import flax.linen as nn
import jax
import jax.nn
import jax.numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import Float, Array, Int


# ------------------------------ type defs ------------------------
class TFTInput(PyTreeNode):
    static: Int[Array, "batch s"]
    known_real: Float[Array, "batch t k"]
    known_categorical: Int[Array, "batch t c"] | None = None
    observed: Float[Array, "batch t o"] | None = None


class TFTEmbeddings(PyTreeNode):
    static: Float[Array, "batch s n"]
    known: Float[Array, "batch n r"]
    observed: Float[Array, "batch n o"] | None = None


class ContextEnrichmentInputs(PyTreeNode):
    history_lstm: jnp.ndarray
    future_lstm: jnp.ndarray
    historical_features: jnp.ndarray
    future_features: jnp.ndarray
    static_context: jnp.ndarray


class ContextInput(PyTreeNode):
    inputs: Float[Array, "batch n"]
    context: Float[Array, "batch n n k"]


class StaticContext(PyTreeNode):
    enrichment: Float[Array, "batch n"]
    state_h: Float[Array, "batch n"]
    state_c: Float[Array, "batch n"]
    vector: Float[Array, "batch n"]


# ----------------------------------------------


class TemporalFusionTransformer(nn.Module):
    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    hidden_layer_size: int
    num_attention_heads: int
    quantiles: List[float]
    num_stacks: int = 1
    dropout_rate: float = 0.1
    output_size: int = 1

    @nn.compact
    def __call__(self, inputs: TFTInput, **kwargs) -> Float[Array, "batch t n"]:
        embeddings = TFTInputEmbedding(
            static_categories_size=self.static_categories_sizes,
            known_categories_size=self.known_categories_sizes,
            hidden_layer_size=self.hidden_layer_size,
        )(inputs)

        # Isolate known and observed historical inputs.
        historical_indexes = jnp.arange(self.num_encoder_steps)
        if inputs.observed is not None:
            historical_inputs = jnp.concatenate(
                [
                    jnp.take(embeddings.known, historical_indexes, axis=1),
                    jnp.take(embeddings.observed, historical_indexes, axis=1),
                ],
                axis=-1,
            )
        else:
            historical_inputs = jnp.take(embeddings.known, historical_indexes, axis=1)

        static_context = StaticCovariatesEncoder(
            hidden_layer_size=self.hidden_layer_size, dropout_rate=self.dropout_rate
        )(embeddings.static)

        total_time_steps = jnp.shape(embeddings.known)[1]

        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps : total_time_steps]
        historical_features, historical_flags, _ = VariableSelection(
            hidden_layer_size=self.hidden_layer_size, dropout_rate=self.dropout_rate
        )(ContextInput(inputs=historical_inputs, context=static_context.enrichment))

        future_features, future_flags, _ = VariableSelection(
            hidden_layer_size=self.hidden_layer_size, dropout_rate=self.dropout_rate
        )(ContextInput(inputs=future_inputs, context=static_context.enrichment))

        state_c, state_h, history_lstm = nn.LSTMCell()(
            [static_context.state_c, static_context.state_h],
            historical_features,
        )
        _, _, future_lstm = self.future_features_lstm(
            [state_c, state_h],
            future_features,
        )

        enriched, temporal_features = ContextEnrichment(
            hidden_layer_size=self.hidden_layer_size, dropout_rate=self.dropout_rate
        )(
            ContextEnrichmentInputs(
                history_lstm=history_lstm,
                future_lstm=future_lstm,
                historical_features=historical_features,
                future_features=future_features,
                static_context=static_context.vector,
            ),
        )
        encoder_in = enriched

        for i in range(self.num_stacks):
            encoder_out = EncoderBlock(
                num_attention_heads=self.num_attention_heads,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
            )(encoder_in)
            encoder_out = nn.LayerNorm(encoder_out + temporal_features)
            encoder_in = encoder_out

        outputs = TimeDistributed(nn.Dense(self.output_size * len(self.quantiles)))(
            encoder_in[:, self.num_encoder_steps : total_time_steps]
        )
        return outputs


# -------------------------------------------------------------------


class TFTInputEmbedding(nn.Module):
    static_categories_size: Int[Array, "s"]
    known_categories_size: Int[Array, "rc"]
    hidden_layer_size: int

    @nn.compact
    def __call__(self, inputs: TFTInput, **kwargs) -> TFTEmbeddings:
        static_input_embeddings = []
        known_real_inputs_embeddings = []

        for i, size in enumerate(self.static_categories_size):
            static_input_embeddings.append(
                nn.Embed(size, self.hidden_layer_size)(inputs.static[:, i])
            )

        static_input_embeddings = jnp.stack(static_input_embeddings, axis=1)

        num_known_real_inputs = inputs.known_real.shape[-1]

        for i in range(num_known_real_inputs):
            known_real_inputs_embeddings.append(
                TimeDistributed(nn.Dense(self.hidden_layer_size))(
                    inputs.known_real[..., i, jnp.newaxis]
                )
            )

        if inputs.observed is not None:
            num_observed_inputs = inputs.observed.shape[-1]
            observed_input_embeddings = []
            for i in range(num_observed_inputs):
                observed_input_embeddings.append(
                    TimeDistributed(nn.Dense(self.hidden_layer_size))(
                        inputs.observed[..., i, jnp.newaxis]
                    )
                )

            observed_input_embeddings = jnp.stack(observed_input_embeddings, axis=-1)
        else:
            observed_input_embeddings = None

        if len(self.known_categories_size) != 0:
            known_categorical_inputs_embeddings = []
            for i, size in enumerate(self.known_categories_size):
                known_categorical_inputs_embeddings.append(
                    nn.Embed(size, self.hidden_layer_size)(
                        inputs.known_categorical[..., i]
                    )
                )
            known_inputs_embeddings = jnp.concatenate(
                [
                    jnp.stack(known_real_inputs_embeddings, axis=-1),
                    jnp.stack(known_categorical_inputs_embeddings, axis=-1),
                ],
                axis=-1,
            )

        else:
            known_inputs_embeddings = jnp.stack(known_real_inputs_embeddings, axis=-1)

        return TFTEmbeddings(
            static=static_input_embeddings,
            observed=observed_input_embeddings,
            known=known_inputs_embeddings,
        )


class StaticCovariatesEncoder(nn.Module):
    hidden_layer_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs: Float[Array, "batch k n"], **kwargs) -> StaticContext:
        num_static_inputs = inputs.shape[1]

        flat_x = jnp.reshape(inputs, (inputs.shape[0], -1))

        mlp_outputs, _ = GRN(
            self.hidden_layer_size,
            output_size=num_static_inputs,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )(flat_x)

        sparse_weights = jax.nn.softmax(mlp_outputs)[..., jnp.newaxis]

        transformed_embeddings = []

        for i in range(num_static_inputs):
            embeds_i, _ = GRN(
                self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
            )(inputs[:, i][:, jnp.newaxis])
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = jnp.concatenate(transformed_embeddings, axis=1)
        static_context_vector = jnp.sum(sparse_weights * transformed_embeddings, axis=1)

        context_variable_selection, _ = GRN(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )(static_context_vector)
        context_enrichment, _ = GRN(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )(static_context_vector)
        context_state_h, _ = GRN(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )(static_context_vector)
        context_state_c, _ = GRN(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
        )(static_context_vector)
        return StaticContext(
            enrichment=context_enrichment,
            state_h=context_state_h,
            state_c=context_state_c,
            vector=context_variable_selection,
            # weight=sparse_weights,
        )


class VariableSelection(nn.Module):
    hidden_layer_size: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        inputs: ContextInput,
        **kwargs,
    ) -> Tuple[
        Float[Array, "batch k m"],
        Float[Array, "batch k c m"],
        Float[Array, "batch k m"],
    ]:
        context = inputs.context
        inputs = inputs.inputs

        time_steps = inputs.shape[1]
        embedding_dim = inputs.shape[2]
        num_inputs = inputs.shape[3]

        mlp_outputs, static_gate = GRN(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            output_size=num_inputs,
            use_time_distributed=True,
        )(
            ContextInput(
                inputs=jnp.reshape(
                    inputs, [-1, time_steps, embedding_dim * num_inputs]
                ),
                context=context[:, jnp.newaxis],
            )
        )
        sparse_weights = jax.nn.softmax(mlp_outputs)
        sparse_weights = sparse_weights[:, :, jnp.newaxis]

        # Non-linear Processing & weight application
        transformed_embeddings = []
        for i in range(num_inputs):
            embeds_i, _ = GRN(
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
            )(inputs[..., i])
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = jnp.stack(transformed_embeddings, axis=-1)
        temporal_ctx = jnp.sum(sparse_weights * transformed_embeddings, axis=-1)
        return temporal_ctx, sparse_weights, static_gate


class TimeDistributed(nn.Module):
    layer: nn.Dense

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        input_shape = inputs.shape
        batch_size = input_shape[0]
        input_length = input_shape[1]
        inner_input_shape = input_shape[2:]
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = jnp.reshape(inputs, [-1, *inner_input_shape])
        # (num_samples * timesteps, ...)
        y = self.layer(inputs, **kwargs)
        # Reconstruct the output shape by re-splitting the 0th dimension
        # back into (num_samples, timesteps, ...)
        # We use batch_size when available so that the 0th dimension is
        # set in the static shape of the reshaped output
        new_inner_shape = y.shape[1:]
        y = jnp.reshape(y, [batch_size, input_length, *new_inner_shape])
        return y


class GLU(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool

    @nn.compact
    def __call__(
        self,
        inputs: Float[Array, "batch time_steps n"] | Float[Array, "batch n"],
        **kwargs,
    ) -> Tuple[
        Float[Array, "batch time_steps n"] | Float[Array, "batch m"],
        Float[Array, "batch time_steps n"] | Float[Array, "batch m"],
    ]:
        dense = nn.Dense(self.hidden_layer_size)
        pre_activation = nn.Dense(self.hidden_layer_size)
        activation = nn.sigmoid

        if self.use_time_distributed:
            dense = TimeDistributed(dense)
            pre_activation = TimeDistributed(pre_activation)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=True)(inputs)
        x_pre_activation = dense(x)
        x_gated = activation(pre_activation(x))
        x = x_pre_activation * x_gated
        return x, x_gated


class GRN(nn.Module):
    hidden_layer_size: int
    dropout_rate: float
    use_time_distributed: bool
    output_size: int | None = None

    @nn.compact
    def __call__(
        self, inputs: Float[Array, "batch n"] | ContextInput, **kwargs
    ) -> Tuple[Float[Array, "batch n"], Float[Array, "batch n"]]:
        if self.output_size is not None:
            skip_connection = nn.Dense(self.output_size)
            if self.use_time_distributed:
                skip_connection = TimeDistributed(skip_connection)
        else:
            skip_connection = Identity()

        pre_elu_dense = nn.Dense(self.hidden_layer_size)
        dense = nn.Dense(self.hidden_layer_size)

        if self.use_time_distributed:
            pre_elu_dense = TimeDistributed(pre_elu_dense)
            dense = TimeDistributed(dense)

        if isinstance(inputs, ContextInput):
            context_dense = nn.Dense(self.hidden_layer_size)
            if self.use_time_distributed:
                context_dense = TimeDistributed(context_dense)

            x_skip = skip_connection(inputs.inputs)
            x = pre_elu_dense(inputs.inputs)
            x = x + context_dense(inputs.context)
        else:
            x_skip = skip_connection(inputs)
            x = pre_elu_dense(inputs)

        x = jax.nn.elu(x)
        x = dense(x)
        x, gate = GLU(
            hidden_layer_size=self.output_size or self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=self.use_time_distributed,
        )(x)
        x = nn.LayerNorm()(x + x_skip)
        return x, gate


class EncoderBlock(nn.Module):
    num_attention_heads: int
    hidden_layer_size: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        inputs: Float[Array, "batch time n"],
        **kwargs,
    ) -> Float[Array, "batch time n"]:
        d_k = self.hidden_layer_size // self.num_attention_heads

        x = nn.SelfAttention(self.num_attention_heads, qkv_features=d_k)(
            inputs, mask=nn.make_causal_mask(inputs)
        )
        x, _ = GLU(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(x)
        x = nn.LayerNorm()(x + inputs)
        # Nonlinear processing on outputs
        decoded, _ = GRN(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(x)
        # Final skip connection
        decoded, _ = GLU(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(decoded)
        return decoded


class ContextEnrichment(nn.Module):
    hidden_layer_size: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self, inputs: ContextEnrichmentInputs, **kwargs
    ) -> Tuple[Float[Array, "batch steps+time n"], Float[Array, "batch steps+time n"]]:
        lstm_outputs = jnp.concatenate(
            [inputs.history_lstm, inputs.future_lstm], axis=1
        )
        input_embeddings = jnp.concatenate(
            [inputs.historical_features, inputs.future_features], axis=1
        )
        lstm_outputs, _ = GLU(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(lstm_outputs)
        temporal_features = nn.LayerNorm()(lstm_outputs + input_embeddings)
        # Static enrichment layers
        expanded_static_context = inputs.static_context[:, jnp.newaxis]

        enriched, _ = GRN(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )(
            ContextInput(
                inputs=temporal_features,
                context=expanded_static_context,
            ),
        )
        return enriched, temporal_features


class Identity(nn.Module):
    @nn.compact
    def __call__(self, inputs, **kwargs):
        return inputs
