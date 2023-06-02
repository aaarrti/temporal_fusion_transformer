from __future__ import annotations

from typing import NamedTuple, Mapping, Tuple, List

import keras.layers as layers
import tensorflow as tf
from keras.utils.tf_utils import can_jit_compile

from temporal_fusion_transformer.experiments.experiment import ExperimentConfig


PRNG_SEED = 28

# TODO docstring, references


class ModelConfig(NamedTuple):
    """
    Attributes
    ----------
    """

    dropout_rate: float = 0.1
    hidden_layer_size: int = 5
    num_attention_heads: int = 4
    output_size: int = 1
    return_attentions: bool = False
    quantiles = [0.1, 0.5, 0.9]


def make_temporal_fusion_transformer(
    model_config: ModelConfig | Mapping[str, int],
    experiment_config: ExperimentConfig | Mapping[str, int],
) -> "TemporalFusionTransformer":
    if isinstance(model_config, Mapping):
        model_config = ModelConfig(**model_config)

    model_config = model_config._asdict()
    if isinstance(experiment_config, Mapping):
        experiment_config = ExperimentConfig(**experiment_config)

    experiment_config = experiment_config._asdict()
    return TemporalFusionTransformer(**model_config, **experiment_config)


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

    @property
    def shape(self) -> tf.TensorShape:
        return tf.shape(self.inputs)


class AttentionComponents(tf.experimental.ExtensionType):
    decoder_self_attn: tf.Tensor
    static_flags: tf.Tensor
    historical_flags: tf.Tensor
    future_flags: tf.Tensor


class TFTInputs(tf.experimental.BatchableExtensionType):
    """
    A convenience container for all different types of outputs fort TFT model.
    Known can be categorical and time varying, e.g, some experiments encode day of the week as real,
    other as categorical.

    Attributes
    ----------

    static:
        Static inputs don't change over time, but are entity-related, e.g, location of the store.
        Static inputs can be categorical only, with shape (batch, num_static_categories).
    known_real:
        shape = (batch, time_steps, n)

    known_categorical:
        shape = (batch, time_steps, m)
    observed:
        Observed (aka unknwown) inputs can be only time varying. They are not known until the observation was made.
        Must have shape (batch, time_steps, num_observed_inputs)
    """

    static: tf.Tensor
    known_real: tf.Tensor
    known_categorical: tf.Tensor
    observed: tf.Tensor

    @property
    def shape(self) -> "TFTInputShape":
        return TFTInputShape(
            static=tf.shape(self.static),
            known_real=tf.shape(self.known_real),
            known_categorical=tf.shape(self.known_categorical),
            observed=tf.shape(self.observed),
        )


class TFTInputShape(tf.experimental.ExtensionType):
    """
    A convenience container to represent all shapes of TFTInputs during model tracing phase.

    Attributes
    ----------

    static:
        [batch, num_inputs]
    known_real:
        [batch, time steps, num_inputs]
    known_categorical:
        [batch, num_inputs]
    observed:
        [batch, time steps, num_inputs]
    """

    static: tf.TensorShape
    known_real: tf.TensorShape
    known_categorical: tf.TensorShape
    observed: tf.TensorShape


class TFTEmbeddedInputs(tf.experimental.ExtensionType):
    static: tf.Tensor
    known: tf.Tensor
    observed: tf.Tensor


class DecoderInputs(tf.experimental.ExtensionType):
    lstm_outputs: tf.Tensor
    input_embeddings: tf.Tensor
    context_vector: tf.Tensor


class StaticContext(tf.experimental.ExtensionType):
    vector: tf.Tensor
    enrichment: tf.Tensor
    state_h: tf.Tensor
    state_c: tf.Tensor
    weights: tf.Tensor


# ------------------------------ base layers --------------------------------


class GatedLinearUnit(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate, seed=PRNG_SEED)
        self.dense_1 = layers.Dense(hidden_layer_size)
        self.dense_2 = layers.Dense(hidden_layer_size, activation="sigmoid")

    def call(self, inputs: tf.Tensor, **kwargs):
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

    def call(self, inputs: tf.Tensor | ContextAwareInputs, **kwargs):
        """
        Parameters
        ----------
        inputs:
            If provided instance of ContextAwareInputs, will add inputs.context to in inputs.inputs,
            before applying gate.

        kwargs:
            Unused.

        Returns
        -------

        """
        if isinstance(inputs, ContextAwareInputs):
            x = self.dense_1(inputs.inputs)
            x = x + self.addition_context_dense(inputs.context)
        else:
            x = self.dense_1(inputs)
        x = tf.nn.elu(x)
        x = self.dense_2(x)
        return x


class GatedResidualNetworkBlock(layers.Layer):
    """Applies the gated residual network (GRN) as defined in paper."""

    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        output_size: int | None = None,
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
    ):
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
        dropout_rate,
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


class TemporalGatedResidualNetworkBlock(GatedResidualNetworkBlock):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float = 0,
        output_size: int | None = None,
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
        static_categories_size: List[int],
        known_categories_size: List[int],
        hidden_layer_size: int,
    ):
        super().__init__(name="tft_input_embedding")
        self.static_categories_size = static_categories_size
        self.known_categories_size = known_categories_size
        self.hidden_layer_size = hidden_layer_size
        self.num_static_inputs = len(self.static_categories_size)
        self.num_known_categorical_inputs = len(self.known_categories_size)

    def build(self, input_shape: TFTInputShape):
        if len(input_shape.static) != 2:
            raise ValueError(
                f"inputs.static must be a 2D tensor, but found shape {input_shape.static}"
            )
        if len(input_shape.known_categorical) != 3:
            raise ValueError(
                f"inputs.known_categorical must be a 3D tensor, but found shape {input_shape.known_categorical}"
            )
        if len(input_shape.known_real) != 3:
            raise ValueError(
                f"inputs.known_real must be a 2D tensor, but found shape {input_shape.known_real}"
            )
        if len(input_shape.observed) != 3:
            raise ValueError(
                f"inputs.observed must be a 2D tensor, but found shape {input_shape.observed}"
            )

        self.num_known_real = input_shape.known_real[-1]
        self.num_observed = input_shape.observed[-1]
        time_steps = input_shape.observed[1]

        self.static_inputs_embedding = [
            layers.Embedding(
                size, self.hidden_layer_size, input_length=time_steps, dtype=tf.float32
            )
            for category, size in enumerate(self.static_categories_size)
        ]
        self.known_categorical_embedding = [
            layers.Embedding(
                size, self.hidden_layer_size, input_length=time_steps, dtype=tf.float32
            )
            for category, size in enumerate(self.known_categories_size)
        ]
        self.known_real_projection = [
            layers.TimeDistributed(layers.Dense(self.hidden_layer_size))
            for _ in range(self.num_known_real)
        ]
        self.observed_projection = [
            layers.TimeDistributed(layers.Dense(self.hidden_layer_size))
            for _ in range(self.num_observed)
        ]

    def call(self, inputs: TFTInputs, *args, **kwargs):
        static = [
            self.static_inputs_embedding[i](inputs.static[:, i])
            for i in range(self.num_static_inputs)
        ]
        observed = [
            self.observed_projection[i](tf.expand_dims(inputs.observed[..., i], -1))
            for i in range(self.num_observed)
        ]
        known_real = [
            self.known_real_projection[i](tf.expand_dims(inputs.known_real[..., i], -1))
            for i in range(self.num_known_real)
        ]
        known_categorical = [
            self.known_categorical_embedding[i](inputs.known_categorical[..., i])
            for i in range(self.num_known_categorical_inputs)
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
        hidden_layer_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        self.num_static = input_shape[1]

        self.grn = GatedResidualNetworkBlock(
            self.hidden_layer_size,
            output_size=self.num_static,
            dropout_rate=self.dropout_rate,
        )
        self.grn_blocks = [
            GatedResidualNetworkBlock(
                self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_static)
        ]

    def call(self, inputs: tf.Tensor, **kwargs):
        flat_x = flatten_over_batch(inputs)
        mlp_outputs, _ = self.grn(flat_x)
        sparse_weights = tf.nn.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        transformed_embedding = []
        for i in tf.range(self.num_static):
            e, _ = self.grn_blocks[i](inputs[:, i : i + 1, :])
            transformed_embedding.append(e)

        transformed_embedding = tf.concat(transformed_embedding, axis=1)
        static_vec = tf.reduce_sum(sparse_weights * transformed_embedding, axis=1)
        return static_vec, sparse_weights


class StaticCovariatesEncoder(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        self.variable_selection = StaticVariableSelectionNetwork(
            self.hidden_layer_size, self.dropout_rate
        )

        self.context_selection = GatedResidualNetworkBlock(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_enrichment = GatedResidualNetworkBlock(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_state_h = GatedResidualNetworkBlock(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.context_state_c = GatedResidualNetworkBlock(
            hidden_layer_size,
            dropout_rate=dropout_rate,
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        # I just followed the naming in original implementation,
        # but they are kinda useless.
        variable_selection, sparse_weights = self.variable_selection(inputs)
        context_variable_selection, _ = self.context_selection(variable_selection)
        context_enrichment, _ = self.context_enrichment(variable_selection)
        context_state_h, _ = self.context_state_h(variable_selection)
        context_state_c, _ = self.context_state_c(variable_selection)

        return StaticContext(
            enrichment=context_enrichment,
            state_h=context_state_h,
            state_c=context_state_c,
            vector=context_variable_selection,
            weights=sparse_weights,
        )


# --------------------------- temporal input components ---------------------------------------


class TemporalVariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate=0,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        _, self.time_steps, self.embedding_dim, self.num_inputs = tf.unstack(
            input_shape
        )

        self.grn = TemporalGatedResidualNetworkBlock(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            output_size=self.num_inputs,
        )
        self.grn_blocks = [
            TemporalGatedResidualNetworkBlock(
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_inputs)
        ]

    def call(
        self, inputs: ContextAwareInputs, *args, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        context = inputs.context
        inputs = inputs.inputs

        mlp_outputs, static_gate = self.grn(
            ContextAwareInputs(
                inputs=tf.reshape(
                    inputs, [-1, self.time_steps, self.embedding_dim * self.num_inputs]
                ),
                context=tf.expand_dims(context, axis=1),
            )
        )
        sparse_weights = tf.nn.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=2)

        # Non-linear Processing & weight application
        transformed_embedding = []
        for i in range(self.num_inputs):
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
        dropout_rate: float,
    ):
        super().__init__(name="temporal_fusion_decoder")

        d_k = hidden_layer_size // num_attention_heads
        self.attn_layer = layers.MultiHeadAttention(
            num_heads=num_attention_heads, key_dim=d_k
        )

        self.glu_1 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)
        self.glu_2 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)
        self.glu_3 = TemporalGatedLinearUnit(hidden_layer_size, dropout_rate)

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.layer_norm_3 = layers.LayerNormalization()

        self.grn_1 = TemporalGatedResidualNetworkBlock(hidden_layer_size, dropout_rate)
        self.grn_2 = TemporalGatedResidualNetworkBlock(hidden_layer_size, dropout_rate)

    def call(self, inputs: DecoderInputs, **kwargs):
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
        x, self_attn = self.attn_layer(
            enriched,
            enriched,
            enriched,
            attention_mask=mask,
            return_attention_scores=True,
        )
        x, _ = self.glu_2(x)
        x = self.layer_norm_2(x + enriched)

        # Nonlinear processing on outputs
        decoded, _ = self.grn_2(x)

        # Final skip connection
        decoded, _ = self.glu_3(decoded)
        decoded = self.layer_norm_3(decoded + temporal_features)
        # return DecoderOutputs(logits=logits, attention=self_att)
        return decoded, self_attn


# ------------------------------- actual model ---------------------------------------------------


class TemporalFusionTransformer(tf.keras.Model):
    def __init__(
        self,
        static_categories_sizes: List[int],  # dataset depended
        known_categories_sizes: List[int],  # dataset depended
        num_encoder_steps: int,  # user depended
        dropout_rate: float,
        hidden_layer_size: int,
        num_attention_heads: int,
        quantiles: List[int],
        output_size: int,
        return_attentions: bool = False,
    ):
        """
        Parameters
        ----------
        static_categories_sizes:
            List with maximum value for each category of static inputs in order.
        known_categories_sizes:
            List with maximum value for each category of known categorical inputs in order.
        num_encoder_steps:
            Number of time steps, which will be considered as past.
        dropout_rate:
            Passed to tf.keras.layer.Dropout(dropout_rate).
        hidden_layer_size:
            Latent space dimensionality.
        num_attention_heads:
            Number of attention heads to use for multi-head attention.
        quantiles
        output_size
        return_attentions
        """
        super().__init__()
        self.quantiles = quantiles
        self.output_size = output_size
        self.num_encoder_steps = num_encoder_steps
        self.hidden_layer_size = hidden_layer_size
        self.static_categories_sizes = static_categories_sizes
        self.known_categories_sizes = known_categories_sizes
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.return_attentions = return_attentions
        self.input_embeds = TFTInputEmbedding(
            static_categories_size=self.static_categories_sizes,
            known_categories_size=known_categories_sizes,
            hidden_layer_size=hidden_layer_size,
        )
        self.static_encoder = StaticCovariatesEncoder(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.historical_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        self.future_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
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
            dropout_rate=dropout_rate,
        )
        self.output_projection = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_size * len(self.quantiles))
        )

    def call(self, inputs: TFTInputs | Mapping[str, tf.Tensor], **kwargs):
        """

        Parameters
        ----------
        inputs:
            pre-processed TFT model inputs:
                - static: (batch, num_static_categories)
                - known_categorical: (batch, time_steps, num_known_categories)
                - known_real: (batch, time_steps)
                - observed: (batch, time_steps)
            Note, that categories are stacked over last axis. E.g., in case of 1 static input,
            static.shape must be (batch, 1). Alternatively, can provide dict-like structure, which will then
            converted to corresponding extension type.

        Returns
        -------

        retval:
            - if return_attentions==False, then return tensor of shape (batch, time_steps - num_encoder_steps, len(quantiles))
            - if return_attentions==False, return 2-tuple, with additional second element containing AttentionComponents instance.

        """
        embeddings: TFTEmbeddedInputs = self.input_embeds(inputs)
        # Isolate known and observed historical inputs.
        historical_inputs = tf.concat(
            [
                embeddings.known[:, : self.num_encoder_steps],
                embeddings.observed[:, : self.num_encoder_steps],
            ],
            axis=-1,
        )
        static_context: StaticContext = self.static_encoder(embeddings.static)

        # Isolate only known future inputs.
        future_inputs = embeddings.known[:, self.num_encoder_steps :]
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
            lstm_outputs=tf.concat([history_lstm, future_lstm], axis=1),
            input_embeddings=tf.concat([historical_features, future_features], axis=1),
            context_vector=static_context.vector,
        )
        decoder_outputs, self_attn = self.temporal_decoder(decoder_in)
        outputs = self.output_projection(
            decoder_outputs[..., self.num_encoder_steps :, :]
        )
        if self.return_attentions:
            attn = AttentionComponents(
                decoder_self_attn=self_attn,
                static_flags=static_context.weights[..., 0],
                historical_flags=historical_flags[..., 0, :],
                future_flags=future_flags[..., 0, :],
            )
            outputs = (outputs, attn)
        return outputs
