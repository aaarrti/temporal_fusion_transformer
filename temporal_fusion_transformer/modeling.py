from __future__ import annotations

import functools
from typing import NamedTuple, Mapping, Tuple, List

import keras.layers as layers
import tensorflow as tf
from keras.utils.tf_utils import can_jit_compile


# TODO replace lists with tensor arrays
# TODO improve naming
# TODO implement from_config to to_config
PRNG_SEED = 28


class TFTConfig(NamedTuple):
    """
    Attributes
    ----------

    num_heads:
        Number of hea ds for interpretable multi-head attention.
    num_stacks:
        Number of self-attention layers to apply.
    num_encoder_steps:
        Size of LSTM encoder -- i.e. number of past time steps before forecast date to use.
    """

    #     quantiles = [0.1, 0.5, 0.9]
    hidden_layer_size: int
    dropout_rate: int
    num_heads: int
    num_stacks: int
    num_encoder_steps: int
    input_size: int
    output_size: int
    time_steps: int
    num_static_inputs: int


def make_temporal_fusion_transformer(
    config: TFTConfig | Mapping[str, int]
) -> "TemporalFusionTransformer":
    if isinstance(config, Mapping):
        config: TFTConfig = TFTConfig(**config)
    return TemporalFusionTransformer(
        num_static_inputs=config.num_static_inputs,
        hidden_layer_size=config.hidden_layer_size,
        num_attention_heads=config.num_heads,
        encoder_steps=None,  # FIXME
        dropout_rate=config.dropout_rate,
    )


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def make_causal_attention_mask(self_attn_inputs: tf.Tensor) -> tf.Tensor:
    len_s = tf.shape(self_attn_inputs)[1]
    bs = tf.shape(self_attn_inputs)[:1]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def flatten_over_batch(arr: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(arr)[0]
    return tf.reshape(arr, (batch_size, -1))


# -------------------------------extension types -----------------------------


class ContextAwareInputs(tf.experimental.BatchableExtensionType):
    context: tf.Tensor
    inputs: tf.Tensor


class AttentionComponents(tf.experimental.ExtensionType):
    decoder_self_attn: tf.Tensor
    static_flags: tf.Tensor
    historical_flags: tf.Tensor
    future_flags: tf.Tensor


# target and observed are always real values
# static and known can be categorical and real valued
# x.shape = (100, 192, 5), y.shape = (100, 24, 1)
# where 5 are all inputs stacked together, not in {InputTypes.ID, InputTypes.TIME} !!
# categorical need to be embedded, time varying must be linearly mapped over time axes (dense + time varying layer)
class TFTInputs(tf.experimental.BatchableExtensionType):
    # e.g. location of the store.
    # static means they dont change over time,
    # but are entity-related.
    # outer most dim is number of such inputs.
    # static can be categorical only
    static: Tuple[tf.Tensor, ...]
    # known can be categorical and time varying
    # e.g., day of the month.
    # Some experiments encode day of the week as real, other as categorical.
    known_real: Tuple[tf.Tensor, ...]
    # e.g., like national holidays
    known_categorical: Tuple[tf.Tensor, ...]
    # observed (aka unknwown) can be only time varying
    # observed are not known until the observation was made
    observed: Tuple[tf.Tensor, ...]
    # Each of those is a 2D tensor [BATCH, TIME_STEP].


class TFTEmbeddedInputs(tf.experimental.ExtensionType):
    static: tf.Tensor
    known: tf.Tensor
    observed: tf.Tensor


class DecoderInputs(tf.experimental.ExtensionType):
    lstm_outputs: tf.Tensor
    input_embeddings: tf.Tensor
    context_vector: tf.Tensor


class DecoderOutputs(tf.experimental.ExtensionType):
    logits: tf.Tensor
    attention: tf.Tensor


class StaticContext(tf.experimental.ExtensionType):
    variable_selection: tf.Tensor
    sparse_weights: tf.Tensor
    enrichment: tf.Tensor
    state_h: tf.Tensor
    state_c: tf.Tensor


class TFTOutputs(tf.experimental.ExtensionType):
    logits: tf.Tensor
    attention: AttentionComponents


# ------------------------------ base layers --------------------------------


class GatedLinearUnit(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate, seed=PRNG_SEED)
        self.dense_1 = layers.Dense(hidden_layer_size)
        self.dense_2 = layers.Dense(hidden_layer_size, activation="sigmoid")

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dropout(inputs)
        x_pre_activation = self.dense_1(x)
        x_gated = self.dense_2(x)
        x = x_pre_activation * x_gated
        return x, x_gated


class FeedForwardNetwork(layers.Layer):
    def __init__(self, hidden_layer_size: int):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dense_1 = layers.Dense(hidden_layer_size)
        self.addition_context_dense = layers.Dense(hidden_layer_size, use_bias=False)
        self.dense_2 = layers.Dense(hidden_layer_size)
        self.elu = layers.ELU()

    def call(self, inputs: tf.Tensor | ContextAwareInputs, **kwargs) -> tf.Tensor:
        if isinstance(inputs, ContextAwareInputs):
            x = self.dense_1(inputs.inputs)
            x = x + self.addition_context_dense(inputs.context)
        else:
            x = self.dense_1(inputs)
        x = self.elu(x)
        x = self.dense_2(x)
        return x


class GatedResidualNetwork(layers.Layer):
    """Applies the gated residual network (GRN) as defined in paper."""

    def __init__(
        self,
        hidden_layer_size: int,
        output_size: int | None = None,
        dropout_rate: float = 0,
    ):
        super().__init__()

        # Setup skip connection.
        if output_size is None:
            output_size = hidden_layer_size
            skip_connection = layers.Identity()
        else:
            skip_connection = layers.Dense(output_size)
        self.skip_connection = skip_connection
        # Setup feedforward network.
        self.ffn = FeedForwardNetwork(hidden_layer_size)
        # Apply gating layer.
        self.glu = GatedLinearUnit(
            output_size or hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.layer_norm = layers.LayerNormalization()
        # Save configurations.
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

    def call(
        self,
        inputs: tf.Tensor | ContextAwareInputs,
        **kwargs,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, ContextAwareInputs):
            x_skip = self.skip_connection(inputs.inputs)
        else:
            x_skip = self.skip_connection(inputs)
        x = self.ffn(inputs)
        x, gate = self.glu(x)
        x = self.layer_norm(x + x_skip)
        return x, gate


# --------------------------- temporal layers -------------------------------


class TemporalGatedLinearUnit(GatedLinearUnit):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate=0,
    ):
        super().__init__(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.dense_1 = layers.TimeDistributed(self.dense_1)
        self.dense_2 = layers.TimeDistributed(self.dense_2)


class TemporalFeedForwardNetwork(FeedForwardNetwork):
    def __init__(self, hidden_layer_size: int):
        super().__init__(hidden_layer_size)
        self.dense_1 = layers.TimeDistributed(self.dense_1)
        self.dense_2 = layers.TimeDistributed(self.dense_2)


class TemporalGatedResidualNetwork(GatedResidualNetwork):
    def __init__(
        self,
        hidden_layer_size: int,
        output_size: int | None = None,
        dropout_rate: float = 0,
    ):
        super().__init__(hidden_layer_size, output_size, dropout_rate)

        # Setup skip connection.
        if output_size is None:
            skip_connection = layers.Identity()
        else:
            skip_connection = layers.TimeDistributed(layers.Dense(output_size))
        self.skip_connection = skip_connection
        # Setup feedforward network.
        self.ffn = TemporalFeedForwardNetwork(hidden_layer_size)
        # Apply gating layer.
        self.glu = TemporalGatedLinearUnit(
            output_size or hidden_layer_size, dropout_rate=dropout_rate
        )


# ----------------------------------- input embedding ----------------------------------------


class TFTInputEmbedding(layers.Layer):
    def __init__(
        self,
        num_static: int,
        static_categories_size: List[int],
        num_known_real: int,
        num_known_categorical: int,
        known_categories_size: List[int],
        num_observed_inputs: int,
        hidden_layer_size: int,
        time_steps: int,
    ):
        super().__init__(name="tft_input_embedding")
        self.num_static_inputs = num_static
        self.static_categories_size = static_categories_size
        self.num_known_real = num_known_real
        self.num_known_categorical = num_known_categorical
        self.known_categories_size = known_categories_size
        self.num_observed_inputs = num_observed_inputs
        self.static_inputs_embedding = [
            layers.Embedding(
                j, hidden_layer_size, input_length=time_steps, dtype=tf.float32
            )
            for i, j in enumerate(static_categories_size)
        ]
        self.known_categorical_embedding = [
            layers.Embedding(
                j, hidden_layer_size, input_length=time_steps, dtype=tf.float32
            )
            for i, j in enumerate(known_categories_size)
        ]
        self.known_real_projection = [
            layers.TimeDistributed(layers.Dense(hidden_layer_size))
            for _ in range(num_known_real)
        ]
        self.observed_projection = [
            layers.TimeDistributed(layers.Dense(hidden_layer_size))
            for _ in range(num_observed_inputs)
        ]

    def call(self, inputs: TFTInputs, *args, **kwargs) -> TFTEmbeddedInputs:
        # TODO replace with tensor array, or mb replace multiple layers with one wider ???
        static = [
            self.static_inputs_embedding[i](j) for i, j in enumerate(inputs.static)
        ]
        observed = [
            self.observed_projection[i](tf.expand_dims(j, -1))
            for i, j in enumerate(inputs.observed)
        ]
        known_real = [
            self.known_real_projection[i](tf.expand_dims(j, -1))
            for i, j in enumerate(inputs.known_real)
        ]
        known_categorical = [
            self.known_categorical_embedding[i](j)
            for i, j in enumerate(inputs.known_categorical)
        ]
        return TFTEmbeddedInputs(
            static=tf.stack(static, axis=1),
            observed=tf.stack(observed, axis=-1),
            known=tf.stack(known_real + known_categorical, axis=-1),
        )


# ------------------------------ static input components ------------------------------------------


class StaticVariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        num_static: int,
        hidden_layer_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.num_static = num_static
        # Nonlinear transformation with gated residual network.
        self.grn = GatedResidualNetwork(
            hidden_layer_size,
            output_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.grn_blocks = [
            GatedResidualNetwork(
                hidden_layer_size,
                output_size=hidden_layer_size,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_static)
        ]

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        flat_x = flatten_over_batch(inputs)
        mlp_outputs, _ = self.grn(flat_x)
        sparse_weights = tf.nn.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        transformed_embedding = []
        for i in tf.range(self.num_static):
            e, _ = self.grn_blocks[i](inputs[:, i])
            transformed_embedding.append(e)

        transformed_embedding = tf.stack(transformed_embedding, axis=-1)
        static_vec = tf.reduce_sum(sparse_weights * transformed_embedding, axis=-1)
        return static_vec, sparse_weights


class StaticCovariatesEncoder(layers.Layer):
    def __init__(
        self,
        num_static: int,
        hidden_layer_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.variable_selection = StaticVariableSelectionNetwork(
            num_static, hidden_layer_size, dropout_rate
        )
        self.context_variable_selection = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_enrichment = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_state_h = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_state_c = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> StaticContext:
        # I just followed the naming in original implementation,
        # but they are kinda useless.
        variable_selection, sparse_weights = self.variable_selection(inputs)
        context_variable_selection, _ = self.context_variable_selection(
            variable_selection
        )
        context_enrichment, _ = self.context_enrichment(variable_selection)
        context_state_h, _ = self.context_state_h(variable_selection)
        context_state_c, _ = self.context_state_c(variable_selection)

        return StaticContext(
            sparse_weights=sparse_weights,
            enrichment=context_enrichment,
            state_h=context_state_h,
            state_c=context_state_c,
            variable_selection=context_variable_selection,
        )


# --------------------------- temporal input components ---------------------------------------


class TemporalVariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        num_encoder_steps: int,
        hidden_layer_size: int,
        num_inputs: int,
        dropout_rate=0,
    ):
        super().__init__(name="temporal_variable_selection")

        # Variable selection weights.
        self.grn = TemporalGatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            output_size=num_inputs,
        )
        self.grn_blocks = [
            TemporalGatedResidualNetwork(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_inputs)
        ]

    def call(
        self, inputs: ContextAwareInputs, *args, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        context = inputs.context
        inputs = inputs.inputs

        _, time_steps, embedding_dim, num_inputs = tf.unstack(tf.shape(inputs))

        mlp_outputs, static_gate = self.grn(
            ContextAwareInputs(
                inputs=tf.reshape(inputs, [-1, time_steps, embedding_dim * num_inputs]),
                context=tf.expand_dims(context, axis=1),
            )
        )
        sparse_weights = tf.nn.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        # Non-linear Processing & weight application
        transformed_embedding = []
        for i in tf.range(num_inputs):
            grn_output, _ = self.grn_blocks[i](
                inputs[..., i],
            )
            transformed_embedding.append(grn_output)

        transformed_embedding = tf.stack(transformed_embedding, axis=-1)
        temporal_ctx = tf.math.reduce_sum(
            sparse_weights * transformed_embedding, axis=-1
        )
        return temporal_ctx, sparse_weights, static_gate


class TemporalFusionDecoder(layers.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_layer_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__(name="temporal_fusion_decoder")
        self.attn_layer = layers.MultiHeadAttention(
            num_heads=num_attention_heads, key_dim=hidden_layer_size
        )

        self.glu_1 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)
        self.glu_2 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)
        self.glu_3 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.layer_norm_3 = layers.LayerNormalization()

        self.grn_1 = TemporalGatedResidualNetwork(
            hidden_layer_size, output_size, dropout_rate
        )
        self.grn_2 = TemporalGatedResidualNetwork(
            hidden_layer_size, output_size, dropout_rate
        )

    def call(self, inputs: DecoderInputs, **kwargs) -> DecoderOutputs:
        lstm_outputs, _ = self.glu_1(inputs.lstm_outputs)
        temporal_features = self.layer_norm_1(lstm_outputs + inputs.input_embeddings)
        # Static enrichment layers
        expanded_static_context = tf.expand_dims(inputs.context_vector, axis=1)

        enriched, _ = self.grn_1(
            ContextAwareInputs(
                inputs=temporal_features,
                context=expanded_static_context,
            ),
        )
        # Decoder self attention
        mask = make_causal_attention_mask(enriched)
        x, self_att = self.attn_layer(
            enriched, enriched, enriched, mask=mask, return_attention_scores=True
        )
        x, _ = self.glu_2(x)
        x = self.layer_norm_2(x + enriched)

        # Nonlinear processing on outputs
        logits = self.grn_2(x)

        # Final skip connection
        logits, _ = self.glu_3(logits)
        logits = self.layer_norm_3(logits + temporal_features)
        return DecoderOutputs(logits=logits, attention=self_att)


# ------------------------------- actual model ---------------------------------------------------


class TemporalFusionTransformer(tf.keras.Model):
    def __init__(
        self,
        num_static_inputs: int,  # dataset depended
        hidden_layer_size: int,
        static_categories_sizes: List[int],  # dataset depended
        num_known_real_inputs: int,  # dataset depended
        num_known_categorical_inputs,  # dataset depended
        known_categories_sizes: List[int],  # dataset depended
        num_observed_inputs: int,  # dataset depended
        num_time_steps: int,
        num_encoder_steps: int,  # dataset depended
        num_attention_heads: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_encoder_steps = num_encoder_steps
        # tum time steps - 192 - total number of datapoints
        # num encoder steps - 168 - number of already observed datapoints
        # 24 - num of time steps we try to predict
        # output_size - number of columns of type TARGET
        self.input_embeds = TFTInputEmbedding(
            num_static=num_static_inputs,
            static_categories_size=static_categories_sizes,
            num_known_real=num_known_real_inputs,
            num_known_categorical=num_known_categorical_inputs,
            known_categories_size=known_categories_sizes,
            num_observed_inputs=num_observed_inputs,
            hidden_layer_size=hidden_layer_size,
            time_steps=num_time_steps,
        )
        self.static_encoder = StaticCovariatesEncoder(
            num_static=num_static_inputs,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )

        self.historical_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            num_encoder_steps=num_encoder_steps,
            dropout_rate=dropout_rate,
            num_inputs=num_observed_inputs
            + num_known_real_inputs
            + num_known_categorical_inputs,
        )
        self.future_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            num_encoder_steps=num_encoder_steps,
            dropout_rate=dropout_rate,
            num_inputs=num_observed_inputs
            + num_known_real_inputs
            + num_known_categorical_inputs,
        )
        self.lstm_encoder = tf.keras.layers.LSTM(
            hidden_layer_size, return_sequences=True, return_state=True
        )
        self.lstm_decoder = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            num_attention_heads=num_attention_heads,
            hidden_layer_size=hidden_layer_size,
            output_size=num_known_categorical_inputs
            + num_known_real_inputs
            + num_static_inputs,
            dropout_rate=dropout_rate,
        )

    def call(
        self,
        inputs: TFTInputs,
        training: bool | None = None,
        mask: None = None,
    ) -> TFTOutputs:
        embeddings: TFTEmbeddedInputs = self.input_embeds(inputs)
        # Isolate known and observed historical inputs.
        historical_inputs = tf.concat(
            [
                embeddings.known[:, : self.num_encoder_steps],
                embeddings.observed[:, : self.num_encoder_steps],
            ],
            axis=-1,
        )
        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps :]
        static_context = self.static_encoder(embeddings.static)
        historical_features, historical_flags, _ = self.historical_variable_selection(
            ContextAwareInputs(
                inputs=historical_inputs, context=static_context.enrichment
            )
        )

        future_features, future_flags, _ = self.future_variable_selection(
            ContextAwareInputs(inputs=future_inputs, context=static_context.enrichment)
        )

        history_lstm, state_h, state_c = self.lstm_encoder(
            historical_features,
            initial_state=[static_context.state_h, static_context.state_c],
        )

        future_lstm = self.lstm_decoder(
            future_features, initial_state=[state_h, state_c]
        )
        decoder_in = DecoderInputs(
            lstm_output=tf.concat([history_lstm, future_lstm], axis=1),
            input_embeddings=tf.concat([historical_features, future_features], axis=1),
            context_vector=static_context.variable_selection,
        )

        logits = self.temporal_decoder(decoder_in)
        return TFTOutputs(
            logits=logits,
            attention=AttentionComponents(
                decoder_self_attn=logits.attention,
                static_flags=static_context.sparse_weights[Ellipsis, 0],
                historical_flags=historical_flags[Ellipsis, 0, :],
                future_flags=future_flags[Ellipsis, 0, :],
            ),
        )
