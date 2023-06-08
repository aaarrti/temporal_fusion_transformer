from __future__ import annotations

from typing import Tuple, Dict, Sequence, TypeVar, Callable, Union
from operator import attrgetter

import keras.layers as layers
import tensorflow as tf
from keras.utils.tf_utils import can_jit_compile


# -------------------------------extension types -----------------------------

# The extension types are used for few reasons:
# 1. keep all args on the first position, so we don't break Keras API contract.
# 2. Improve static code analysis, in comparison with, e.g., using dictionaries.
# 3. To explicitly separate different inputs, so it is not so easy to mess up, in comparison with, e.g,
#    just stacking all in one tensor as it was done in TF1.


class TFTInputs(tf.experimental.BatchableExtensionType):
    """
    TODO i probably will need to get rid of it, so I can use TPUs
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
    known_categorical: Union[tf.Tensor, None] = None
    observed: Union[tf.Tensor, None] = None

    @property
    def dtype(self):
        return self.static.dtype

    @property
    def shape(self) -> TFTInputShape:
        return TFTInputShape(
            static=self.static.shape,
            known_real=self.known_real.shape,
            known_categorical=map_optional(self.known_categorical, attrgetter("shape")),
            observed=map_optional(self.observed, attrgetter("shape")),
        )

    # class Spec:
    #     def __init__(self, shape: TFTInputShape, dtype=tf.float32):
    #         self.values = tf.TensorSpec(shape, dtype)
    #         self.mask = tf.TensorSpec(shape, tf.bool)
    #
    #     def __repr__(self):
    #         return f"TFTInputs.Spec(shape={self.shape}, dtype={self.dtype})"
    #
    #     shape = property(lambda self: self.values.shape)
    #     dtype = property(lambda self: self.values.dtype)


class TFTInputShape(tf.experimental.BatchableExtensionType):
    static: tf.TensorShape
    known_real: tf.TensorShape
    known_categorical: Union[tf.TensorShape, None]
    observed: Union[tf.TensorShape, None]


class TFTEmbeddings(tf.experimental.BatchableExtensionType):
    """

    A convenience container to represent output of TFTInputEmbedding layer.

    Attributes
    ----------

    static:
        (batch, n_s, hidden_layer_size)
    known:
        (batch, time_steps, hidden_layer_size, n_kr + n_kc)
    observed:
        (batch, time_steps, n_o)

    """

    static: tf.Tensor
    known: tf.Tensor
    observed: Union[tf.Tensor, None]

    @property
    def dtype(self):
        return self.static.dtype

    @property
    def shape(self) -> TFTEmbeddingsShape:
        return TFTEmbeddingsShape(
            static=self.static.shape,
            known=self.known.shape,
            observed=map_optional(self.observed, attrgetter("shape")),
        )


class TFTEmbeddingsShape(tf.experimental.BatchableExtensionType):
    static: tf.TensorShape
    known: tf.TensorShape
    observed: Union[tf.TensorShape, None]


class StaticContext(tf.experimental.BatchableExtensionType):
    """
    A convenience container to represent different outputs of StaticCovariatesEncoder layer.

    Attributes
    ----------

    enrichment:
        has shape (batch_size, hidden_layer_size) and must be used as additional context for GRNs.

    vector:
        has shape (batch_size, hidden_layer_size), must be passed down to temporal decoder, and used  as additional context in GRNs.

    state_c:
        has shape (batch_size, hidden_layer_size) and must be used together with `state_h` as initial context for LSTM cells.

    state_h:
        has shape (batch_size, hidden_layer_size) and must be used together with `state_c` as initial context for LSTM cells.
    """

    vector: tf.Tensor
    enrichment: tf.Tensor
    state_h: tf.Tensor
    state_c: tf.Tensor


class ContextAwareInputs(tf.experimental.BatchableExtensionType):
    """
    A convenience container for providing static context to layer inputs.

    Attributes
    ----------

    context:
        A context vector.
    inputs:
        Tensor, to which context should be applied.
    """

    context: tf.Tensor
    inputs: tf.Tensor

    @property
    def shape(self) -> tf.TensorShape:
        return self.inputs.shape


class DecoderInputs(tf.experimental.BatchableExtensionType):
    """

    Attributes
    ----------

    lstm_outputs:

    input_embeddings:

    context_vector:

    """

    lstm_outputs: tf.Tensor
    input_embeddings: tf.Tensor
    context_vector: tf.Tensor


# ------------------------------- TFT model ---------------------------------------------------


class TemporalFusionTransformer(tf.keras.Model):
    """
    References:
    - Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
        - https://arxiv.org/pdf/1912.09363.pdf
        - https://github.com/google-research/google-research/tree/master/tft

    """

    def __init__(
        self,
        *,
        static_categories_sizes: Sequence[int],  # dataset depended
        known_categories_sizes: Sequence[int],  # dataset depended
        num_encoder_steps: int,
        hidden_layer_size: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
        output_size: int = 1,
        quantiles: Sequence[int] | None = None,
        prng_seed: int = 42,
        name: str = "temporal_fusion_transformer",
        unroll_lstm: bool = False,
        return_attention: bool = False,
        **kwargs,
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
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.
        hidden_layer_size:
            Latent space dimensionality.
        num_attention_heads:
            Number of attention heads to use for multi-head attention.
        quantiles:
        output_size:
        unroll_lstm:
        return_attention:


        name:

        """
        super().__init__(name=name, **kwargs)
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.output_size = output_size
        self.num_encoder_steps = num_encoder_steps
        self.hidden_layer_size = hidden_layer_size
        self.static_categories_sizes = static_categories_sizes
        self.known_categories_sizes = known_categories_sizes
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.prng_seed = prng_seed
        self.unroll_lstm = unroll_lstm
        self.return_attention = return_attention

        self.input_embeds = TFTInputEmbedding(
            static_categories_size=self.static_categories_sizes,
            known_categories_size=known_categories_sizes,
            hidden_layer_size=hidden_layer_size,
        )
        self.static_encoder = StaticCovariatesEncoder(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
        )
        self.historical_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            name="historical_variable_selection",
        )
        self.future_variable_selection = TemporalVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            name="future_variable_selection",
        )
        self.historical_features_lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            return_state=True,
            name="historical_features_lstm",
            unroll=unroll_lstm,
        )
        self.future_features_lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            name="future_features_lstm",
            unroll=unroll_lstm,
        )
        self.temporal_decoder = TemporalFusionDecoder(
            num_attention_heads=num_attention_heads,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
        )
        self.output_projection = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_size * len(self.quantiles))
        )

    def call(self, inputs: TFTInputs, **kwargs):
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
            static.shape must be (batch, 1).

        Returns
        -------

        retval:
            Tensor of shape (batch, time_steps - num_encoder_steps, len(quantiles)).

        """
        embeddings: TFTEmbeddings = self.input_embeds(inputs)
        # Isolate known and observed historical inputs.
        if inputs.observed is not None:
            historical_inputs = tf.concat(
                [
                    embeddings.known[:, : self.num_encoder_steps],
                    embeddings.observed[:, : self.num_encoder_steps],
                ],
                axis=-1,
            )
        else:
            historical_inputs = embeddings.known[:, : self.num_encoder_steps]

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

        history_lstm, state_h, state_c = self.historical_features_lstm(
            historical_features,
            initial_state=[static_context.state_h, static_context.state_c],
        )

        future_lstm = self.future_features_lstm(
            future_features, initial_state=[state_h, state_c]
        )
        decoder_in = DecoderInputs(
            lstm_outputs=tf.concat([history_lstm, future_lstm], axis=1),
            input_embeddings=tf.concat([historical_features, future_features], axis=1),
            context_vector=static_context.vector,
        )
        decoder_outputs = self.temporal_decoder(decoder_in)
        outputs = self.output_projection(
            decoder_outputs[..., self.num_encoder_steps :, :]
        )
        # attention_components = {
        # # Temporal attention weights
        # "decoder_self_attn": self_att,
        # # Static variable selection weights
        # "static_flags": static_weights[Ellipsis, 0],
        # # Variable selection weights of past inputs
        # "historical_flags": historical_flags[Ellipsis, 0, :],
        # # Variable selection weights of future inputs
        # "future_flags": future_flags[Ellipsis, 0, :],
        # }
        return outputs

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "quantiles": self.quantiles,
                "output_size": self.output_size,
                "num_encoder_steps": self.num_encoder_steps,
                "hidden_layer_size": self.hidden_layer_size,
                "static_categories_sizes": self.static_categories_sizes,
                "known_categories_sizes": self.known_categories_sizes,
                "num_attention_heads": self.num_attention_heads,
                "prng_seed": self.prng_seed,
                "unroll_lstm": self.unroll_lstm,
                "return_attention": self.return_attention,
            }
        )
        return config


# ----------------------------------- input embedding ----------------------------------------


class TFTInputEmbedding(layers.Layer):
    def __init__(
        self,
        static_categories_size: Sequence[int],
        known_categories_size: Sequence[int],
        hidden_layer_size: int,
        name: str = "tft_input_embedding",
        **kwargs,
    ):
        """
        Parameters
        ----------
        static_categories_size:
            Sequence of ints describing with max value for each category of static inputs.
            E.g., you have `shop_name` (which can only be "a", "b" or "c") and `location_city`
            (which can only be "A", "B", "C" or "D") as your static inputs, then static_categories_size=[3, 4].
        known_categories_size:
            Sequence of ints describing with max value for each category of known categorical inputs.
            This follows the same principle as `static_categories_size`, with only difference that known inputs can
            change overtime. An example of know categorical input can be day of the week.
        hidden_layer_size:
            Latent space dimensionality.
        name:
            Name for the layer.
        kwargs:
            Standard Keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)
        static_categories_size = static_categories_size
        known_categories_size = known_categories_size
        hidden_layer_size = hidden_layer_size
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
        if not input_shape.known_categorical == tf.TensorShape(None):
            if len(input_shape.known_categorical) != 3:
                raise ValueError(
                    f"inputs.known_categorical must be a 3D tensor, but found shape {input_shape.known_categorical}"
                )
        if len(input_shape.known_real) != 3:
            raise ValueError(
                f"inputs.known_real must be a 2D tensor, but found shape {input_shape.known_real}"
            )
        if not input_shape.observed == tf.TensorShape(None):
            if len(input_shape.observed) != 3:
                raise ValueError(
                    f"inputs.observed must be a 2D tensor, but found shape {input_shape.observed}"
                )

        self.num_known_real_inputs = input_shape.known_real[-1]
        if input_shape.observed == tf.TensorShape(None):
            self.num_observed_inputs = 0
        else:
            self.num_observed_inputs = input_shape.observed[-1]
        num_time_steps = input_shape.known_real[1]

        self.static_inputs_embedding = [
            layers.Embedding(
                size,
                self.hidden_layer_size,
                input_length=num_time_steps,
                # dtype=self.dtype_policy.compute_dtype,
            )
            for size in self.static_categories_size
        ]
        self.known_categorical_embedding = [
            layers.Embedding(
                size,
                self.hidden_layer_size,
                input_length=num_time_steps,
                # dtype=self.dtype_policy.compute_dtype,
            )
            for size in self.known_categories_size
        ]
        self.known_real_dense = [
            layers.TimeDistributed(layers.Dense(self.hidden_layer_size))
            for _ in range(self.num_known_real_inputs)
        ]
        self.observed_dense = [
            layers.TimeDistributed(layers.Dense(self.hidden_layer_size))
            for _ in range(self.num_observed_inputs)
        ]

    def call(self, inputs: TFTInputs, **kwargs) -> TFTEmbeddings:
        """
        This layer project all different inputs into the same latent space.
        Embedding is applied to categorical ones, and real-values ones are
        linearly mapped over `time_steps` axis.

        Parameters
        ----------
        inputs:
            A named tuple-like structure with static, known and observed inputs, with shape:
            - static: (batch, n_s)
            - known_real: (batch, time_steps, n_kr)
            - known_categorical: (batch, time_steps, n_kc)
            - observed: (batch, time_steps, n_o)
        kwargs:
            Standard Keras layers' kwargs.

        Returns
        -------

        embeds:
            Instance of TFTEmbeddings, where static, known and observed inputs all have been projected into latent
            space, with shape:
            - static: (batch, n_s, hidden_layer_size)
            - known: (batch, time_steps, hidden_layer_size, n_kr + n_kc)
            - observed: (batch, time_steps, n_o)

        """
        static_input_embeddings = []

        known_real_inputs_embeddings = []

        for i, layer in enumerate(self.static_inputs_embedding):
            static_input_embeddings.append(layer(inputs.static[:, i]))
        static_input_embeddings = tf.stack(static_input_embeddings, axis=1)

        for i, layer in enumerate(self.known_real_dense):
            known_real_inputs_embeddings.append(
                layer(tf.expand_dims(inputs.known_real[..., i], -1))
            )

        if self.num_observed_inputs != 0:
            observed_input_embeddings = []
            for i, layer in enumerate(self.observed_dense):
                observed_input_embeddings.append(
                    layer(tf.expand_dims(inputs.observed[..., i], axis=-1))
                )
            observed_input_embeddings = tf.stack(observed_input_embeddings, axis=-1)
        else:
            observed_input_embeddings = None

        if self.num_known_categorical_inputs != 0:
            known_categorical_inputs_embeddings = []
            for i, layer in enumerate(self.known_categorical_embedding):
                known_categorical_inputs_embeddings.append(
                    layer(inputs.known_categorical[..., i])
                )
            known_inputs_embeddings = tf.concat(
                [
                    tf.stack(known_real_inputs_embeddings, axis=-1),
                    tf.stack(known_categorical_inputs_embeddings, axis=-1),
                ],
                axis=-1,
            )

        else:
            known_inputs_embeddings = tf.stack(known_real_inputs_embeddings, axis=-1)

        return TFTEmbeddings(
            static=static_input_embeddings,
            observed=observed_input_embeddings,
            known=known_inputs_embeddings,
        )

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "static_categories_size": list(self.static_categories_size),
                "known_categories_size": list(self.known_categories_size),
                "hidden_layer_size": self.hidden_layer_size,
            }
        )
        return config


# ------------------------------ static covariates encoder ------------------------------------------


class StaticCovariatesEncoder(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        prng_seed: int,
        name: str = "static_covariates_encoder",
        **kwargs,
    ):
        """
        Parameters
        ----------
        hidden_layer_size:
            Dimensionality of the latent space.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.
        name:
            Name for the layer.
        kwargs:
            Standard Keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)
        # Save configs.
        hidden_layer_size = hidden_layer_size
        dropout_rate = dropout_rate
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.prng_seed = prng_seed
        # Build sub layers.
        self.context_selection = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=False,
        )
        self.context_enrichment = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=False,
        )
        self.context_state_h = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=False,
        )
        self.context_state_c = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=False,
        )

    def build(self, input_shape: tf.TensorShape):
        self.num_static_inputs = input_shape[1]
        self.grn = GatedResidualNetwork(
            self.hidden_layer_size,
            output_size=self.num_static_inputs,
            dropout_rate=self.dropout_rate,
            prng_seed=self.prng_seed,
            use_time_distributed=False,
        )
        self.grn_blocks = [
            GatedResidualNetwork(
                self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                prng_seed=self.prng_seed,
                use_time_distributed=False,
            )
            for _ in range(self.num_static_inputs)
        ]

    def call(self, inputs: tf.Tensor, **kwargs) -> StaticContext:
        """
        Create a static context out of static input embeddings.
        Static context is a (enrichment) vector, which must be added to other time varying inputs during
        `variable selection` process. Additionally, this layer creates initial state for LSTM cells,
        this way we give them `pre-memory`, and model the fact that static inputs do influence time dependent ones.


        Parameters
        ----------
        inputs:
            Static inputs' embeddings of shape (batch, n, hidden_layer_size).
        kwargs:
            Standard Keras layers' kwargs, unused.

        Returns
        -------

        static_context:
            instance of StaticContext, which layer must be used to enrich time dependent inputs.
            It has a named tuple-like structure with elements:
                - enrichment: has shape (batch_size, hidden_layer_size) and must be used as additional context for GRNs.
                - vector: has shape (batch_size, hidden_layer_size), must be passed down to temporal decoder, and used  as additional context in GRNs.
                - state_c: has shape (batch_size, hidden_layer_size) and must be used together with `state_h` as initial context for LSTM cells.
                - state_h: has shape (batch_size, hidden_layer_size) and must be used together with `state_c` as initial context for LSTM cells.
        """

        flat_x = flatten_over_batch(inputs)

        mlp_outputs, _ = self.grn(flat_x)
        sparse_weights = tf.nn.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        transformed_embeddings = []

        for i, layer in enumerate(self.grn_blocks):
            embeds_i, _ = layer(inputs[:, i : i + 1, :])
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = tf.concat(transformed_embeddings, axis=1)
        static_context_vector = tf.reduce_sum(
            sparse_weights * transformed_embeddings, axis=1
        )
        context_variable_selection, _ = self.context_selection(static_context_vector)
        context_enrichment, _ = self.context_enrichment(static_context_vector)
        context_state_h, _ = self.context_state_h(static_context_vector)
        context_state_c, _ = self.context_state_c(static_context_vector)
        return StaticContext(
            enrichment=context_enrichment,
            state_h=context_state_h,
            state_c=context_state_c,
            vector=context_variable_selection,
        )

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "hidden_layer_size": self.hidden_layer_size,
                "dropout_rate": self.dropout_rate,
                "prng_seed": self.prng_seed,
            }
        )
        return config


# --------------------------------- GLU && GRN --------------------------------


class GatedLinearUnit(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        prng_seed: int,
        use_time_distributed: bool,
        name: str = "glu",
        **kwargs,
    ):
        """
        Parameters
        ----------
        hidden_layer_size:
            Latent space dimensionality.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.

        use_time_distributed:
            Whether to apply across time axis.
        name:
            Name for layer.
        kwargs:
            Standard Keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)
        # Save configurations.
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.prng_seed = prng_seed
        self.use_time_distributed = use_time_distributed
        # Build sub-layers.
        self.dropout = layers.Dropout(dropout_rate, seed=prng_seed)
        self.linear = layers.Dense(hidden_layer_size)
        self.activation = layers.Dense(hidden_layer_size, activation="sigmoid")
        # Apply time steps, if needed.
        if use_time_distributed:
            self.linear = layers.TimeDistributed(self.linear)
            self.activation = layers.TimeDistributed(self.activation)

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies a Gated Linear Unit (GLU) to an input.

        Parameters
        ----------
        inputs:
            Input tensor for gating layer.
        kwargs:
            Standard Keras layers' kwargs, unused.

        Returns
        -------

        retval:
            Tuple of tensors for: (GLU output, gate)

        """
        x = self.dropout(inputs)
        x_pre_activation = self.linear(x)
        x_gated = self.activation(x)
        x = x_pre_activation * x_gated
        return x, x_gated

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "hidden_layer_size": self.hidden_layer_size,
                "dropout_rate": self.dropout_rate,
                "prng_seed": self.prng_seed,
                "use_time_distributed": self.use_time_distributed,
            }
        )
        return config


class GatedResidualNetwork(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        use_time_distributed: bool,
        prng_seed: int,
        output_size: int | None = None,
        name: str = "grn",
        **kwargs,
    ):
        """

        Parameters
        ----------
        hidden_layer_size:
            Latent space dimensionality.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.
        output_size:
            Size of output layer, default=hidden_layer_size.
        use_time_distributed:
            Whether to apply across time axis.
        name:
            Name for layer.
        kwargs:
            Standard Keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)
        # Save configurations.
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.prng_seed = prng_seed
        self.output_size = output_size
        # Setup skip connection.
        if output_size is None:
            output_size = hidden_layer_size
            skip_connection = layers.Identity()
        else:
            skip_connection = layers.Dense(output_size)
            if use_time_distributed:
                skip_connection = layers.TimeDistributed(skip_connection)
        self.skip_connection = skip_connection
        # Setup feedforward network.
        self.pre_elu_dense = layers.Dense(hidden_layer_size)
        self.context_dense = layers.Dense(hidden_layer_size)
        self.linear_dense = layers.Dense(hidden_layer_size)
        # Apply gating layer.
        self.gated_linear_unit = GatedLinearUnit(
            output_size or hidden_layer_size,
            dropout_rate=dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=use_time_distributed,
        )
        self.layer_norm = layers.LayerNormalization()
        if use_time_distributed:
            self.pre_elu_dense = layers.TimeDistributed(self.pre_elu_dense)
            self.context_dense = layers.TimeDistributed(self.context_dense)
            self.linear_dense = layers.TimeDistributed(self.linear_dense)

    def call(
        self,
        inputs: tf.Tensor | ContextAwareInputs,
        **kwargs,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the gated residual network (GRN) as defined in paper.

        References:
        - Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
            - https://arxiv.org/pdf/1912.09363.pdf
            - https://github.com/google-research/google-research/tree/master/tft


        Parameters
        ----------
        inputs:
            Network inputs
        kwargs:
            Standard keras layers' kwargs, unused.

        Returns
        -------
        retval:
            Tuple of tensors for: (GRN output, GLU gate)

        """
        if isinstance(inputs, ContextAwareInputs):
            x_skip = self.skip_connection(inputs.inputs)
            x = self.pre_elu_dense(inputs.inputs)
            x = x + self.context_dense(inputs.context)
        else:
            x_skip = self.skip_connection(inputs)
            x = self.pre_elu_dense(inputs)

        x = tf.nn.elu(x)
        x = self.linear_dense(x)
        x, gate = self.gated_linear_unit(x)
        x = self.layer_norm(x + x_skip)
        return x, gate

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "output_size": self.output_size,
                "hidden_layer_size": self.hidden_layer_size,
                "dropout_rate": self.dropout_rate,
                "use_time_distributed": self.use_time_distributed,
                "prng_seed": self.prng_seed,
            }
        )
        return config


# --------------------------- variable selection network ---------------------------------------


class TemporalVariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        prng_seed: int,
        name: str = "variable_selection",
        **kwargs,
    ):
        """

        Parameters
        ----------
        hidden_layer_size
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.
        name:
            A name for layer.
        kwargs:
            Standard Keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.prng_seed = prng_seed

    def build(self, input_shape: tf.TensorShape):
        self.time_steps = input_shape[1]
        self.embedding_dim = input_shape[2]
        self.num_inputs = input_shape[3]

        self.context_grn = GatedResidualNetwork(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            output_size=self.num_inputs,
            prng_seed=self.prng_seed,
            use_time_distributed=True,
        )

        self.grn_blocks = [
            GatedResidualNetwork(
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                prng_seed=self.prng_seed,
                use_time_distributed=True,
            )
            for _ in range(self.num_inputs)
        ]

    def call(
        self, inputs: ContextAwareInputs, **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        context = inputs.context
        inputs = inputs.inputs

        mlp_outputs, static_gate = self.context_grn(
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
        transformed_embeddings = []
        for i, layer in enumerate(self.grn_blocks):
            embeds_i, _ = layer(inputs[..., i])
            transformed_embeddings.append(embeds_i)

        transformed_embeddings = tf.stack(transformed_embeddings, axis=-1)
        temporal_ctx = tf.math.reduce_sum(
            sparse_weights * transformed_embeddings, axis=-1
        )
        return temporal_ctx, sparse_weights, static_gate

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "hidden_layer_size": self.hidden_layer_size,
                "dropout_rate": self.dropout_rate,
                "prng_seed": self.prng_seed,
            }
        )
        return config


class TemporalFusionDecoder(layers.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_layer_size: int,
        dropout_rate: float,
        prng_seed: int,
        name: str = "temporal_fusion_decoder",
        **kwargs,
    ):
        """

        Parameters
        ----------
        num_attention_heads
        hidden_layer_size
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        prng_seed:
            PRNG seed to be used for keras.layer.Dropout.
        name:
            A name for layer.
        kwargs:
            Standard keras layers' kwargs.
        """
        super().__init__(name=name, **kwargs)

        self.num_attention_heads = num_attention_heads
        self.prng_seed = prng_seed
        self.dropout_rate = dropout_rate
        self.hidden_layer_size = hidden_layer_size

        d_k = hidden_layer_size // num_attention_heads
        self.attn_layer = layers.MultiHeadAttention(
            num_heads=num_attention_heads, key_dim=d_k
        )

        self.glu_1 = GatedLinearUnit(
            hidden_layer_size,
            dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=True,
        )
        self.glu_2 = GatedLinearUnit(
            hidden_layer_size,
            dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=True,
        )
        self.glu_3 = GatedLinearUnit(
            hidden_layer_size,
            dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=True,
        )

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.layer_norm_3 = layers.LayerNormalization()

        self.grn_1 = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=True,
        )
        self.grn_2 = GatedResidualNetwork(
            hidden_layer_size,
            dropout_rate,
            prng_seed=prng_seed,
            use_time_distributed=True,
        )

    def call(self, inputs: DecoderInputs, **kwargs) -> tf.Tensor:
        """

        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
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
        x = self.attn_layer(
            enriched,
            enriched,
            enriched,
            attention_mask=mask,
        )
        x, _ = self.glu_2(x)
        x = self.layer_norm_2(x + enriched)

        # Nonlinear processing on outputs
        decoded, _ = self.grn_2(x)

        # Final skip connection
        decoded, _ = self.glu_3(decoded)
        decoded = self.layer_norm_3(decoded + temporal_features)
        return decoded

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "prng_seed": self.prng_seed,
                "dropout_rate": self.dropout_rate,
                "hidden_layer_size": self.hidden_layer_size,
            }
        )
        return config


# ---------------------- helper functions ---------------------------


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


T = TypeVar("T")
R = TypeVar("R")


def map_optional(optional: T | None, func: Callable[[T], R]) -> R | None:
    if optional is not None:
        return func(optional)
    else:
        return None
