from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Tuple, TypedDict

import keras_core as keras
from keras_core import layers, ops
from toolz import dicttoolz

from temporal_fusion_transformer.src.utils import enumerate_v2

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict

newaxis = None

log = logging.getLogger(__name__)


class TemporalFusionTransformer(keras.Model):
    """
    References
    ----------

    Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/pdf/1912.09363.pdf, https://github.com/google-research/google-research/tree/master/tft

    """

    def __init__(
        self,
        *,
        input_observed_idx,
        input_static_idx,
        input_known_real_idx,
        input_known_categorical_idx,
        static_categories_sizes,
        known_categories_sizes,
        latent_dim,
        dropout_rate,
        num_encoder_steps,
        total_time_steps,
        num_attention_heads,
        num_decoder_blocks,
        num_quantiles,
        attention_dropout_rate=0,
        num_outputs=1,
        return_attention=False,
        unroll: bool = False,
        name="TemporalFusionTransformer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_encoder_steps = num_encoder_steps
        self.total_time_steps = total_time_steps
        self.return_attention = return_attention

        num_known_real_inputs = len(input_known_real_idx)
        num_known_categorical_inputs = len(input_known_categorical_idx)
        num_observed_inputs = len(input_observed_idx)

        self.input_preprocessor = InputPreprocessor(
            input_observed_idx=input_observed_idx,
            input_static_idx=input_static_idx,
            input_known_real_idx=input_known_real_idx,
            input_known_categorical_idx=input_known_categorical_idx,
        )
        self.input_embedding = InputEmbedding(
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=known_categories_sizes,
            num_known_real_inputs=len(input_known_real_idx),
            num_observed_inputs=num_observed_inputs,
            latent_dim=latent_dim,
        )
        self.static_covariates_encoder = StaticCovariatesEncoder(
            dropout_rate=dropout_rate, latent_dim=latent_dim, num_static_inputs=len(static_categories_sizes)
        )
        self.historical_variable_selection = VariableSelectionNetwork(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            num_inputs=num_known_real_inputs + num_known_categorical_inputs + num_observed_inputs,
            num_time_steps=num_encoder_steps,
        )
        self.future_variable_selection = VariableSelectionNetwork(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            num_time_steps=total_time_steps - num_encoder_steps,
            num_inputs=num_known_real_inputs + num_known_categorical_inputs,
        )
        self.historical_lstm = layers.LSTM(latent_dim, return_state=True, return_sequences=True, unroll=unroll)

        self.future_lstm = layers.LSTM(latent_dim, return_sequences=True, unroll=unroll)

        self.output_skip_connection = GatedLinearUnit(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=True,
        )

        self.lstm_skip_connection = GatedLinearUnit(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=True,
        )
        self.temporal_ln = layers.LayerNormalization()

        self.static_context_skip_connection = ContextualGatedResidualNetwork(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=True,
        )

        self.decoder_blocks = [
            DecoderBlock(
                num_attention_heads=num_attention_heads,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
            )
            for _ in range(num_decoder_blocks)
        ]
        self.decoder_ln = [layers.LayerNormalization() for _ in range(num_decoder_blocks)]

        self.output_projection = [layers.TimeDistributed(layers.Dense(num_outputs)) for _ in range(num_quantiles)]

    def __call__(self, inputs: Array, **kwargs) -> Array | TftOutputs:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: Array, **kwargs) -> Array | TftOutputs:
        inputs = self.input_preprocessor(inputs)
        embeddings = self.input_embedding(inputs)
        static_context = self.static_covariates_encoder(embeddings["static"])

        # Isolate known and observed historical inputs.
        historical_inputs = [embeddings["known"][:, : self.num_encoder_steps]]
        if embeddings["observed"].size > 0:
            historical_inputs.append(embeddings["observed"][:, : self.num_encoder_steps])

        historical_inputs = ops.concatenate(historical_inputs, axis=-1)

        # Isolate only known future inputs.
        future_inputs = embeddings["known"][:, self.num_encoder_steps : self.total_time_steps]
        historical_features, historical_flags, _ = self.historical_variable_selection(
            {"inputs": historical_inputs, "context": static_context["enrichment"]}
        )
        future_features, future_flags, _ = self.future_variable_selection(
            {"inputs": future_inputs, "context": static_context["enrichment"]}
        )

        history_lstm, state_h, state_c = self.historical_lstm(
            historical_features,
            initial_state=[static_context["state_h"], static_context["state_c"]],
        )
        future_lstm = self.future_lstm(future_features, initial_state=[state_h, state_c])

        lstm_layer = ops.concatenate([history_lstm, future_lstm], axis=1)
        input_embeddings = ops.concatenate([historical_features, future_features], axis=1)

        lstm_outputs, _ = self.lstm_skip_connection(lstm_layer)
        temporal_features = self.temporal_ln(lstm_outputs + input_embeddings)

        enriched, _ = self.static_context_skip_connection(
            {"inputs": temporal_features, "context": static_context["vector"][:, newaxis]}
        )
        decoder_in = enriched

        for ln, block in zip(self.decoder_ln, self.decoder_blocks):
            decoder_out = block(decoder_in)
            decoder_out = ln(decoder_out + temporal_features)
            decoder_in = decoder_out

        # Final skip connection
        decoded, _ = self.output_skip_connection(decoder_in)

        outputs = []

        for layer in self.output_projection:
            outputs_i = layer(decoded[:, self.num_encoder_steps : self.total_time_steps])
            outputs.append(outputs_i)

        outputs = ops.stack(outputs, axis=-1)

        if self.return_attention:
            return {
                "logits": outputs,
                "historical_flags": historical_flags[..., 0, :],
                "future_flags": future_flags[..., 0, :],
                "static_flags": static_context["weight"][..., 0],
            }
        else:
            return outputs

    @classmethod
    def from_config_dict(cls, config: ConfigDict) -> TemporalFusionTransformer:
        kwargs = {**config.model.to_dict(), **config.data.to_dict()}
        allowed_kwargs = inspect.signature(cls.__init__).parameters
        init_kwargs = dicttoolz.keyfilter(lambda k: k in allowed_kwargs, kwargs)
        return TemporalFusionTransformer(**init_kwargs, num_quantiles=len(config.quantiles))


# -------------------------------------------------------------------------------------------------------------


class InputPreprocessor(layers.Layer):
    def __init__(
        self,
        *,
        input_observed_idx,
        input_static_idx,
        input_known_real_idx,
        input_known_categorical_idx,
        dtype="float32",
        name="input_preprocessor",
        **kwargs,
    ):
        super().__init__(dtype=dtype, name=name, **kwargs)
        self.input_observed_idx = input_observed_idx
        self.input_static_idx = input_static_idx
        self.input_known_real_idx = input_known_real_idx
        self.input_known_categorical_idx = input_known_categorical_idx

    def __call__(self, inputs: Array, **kwargs) -> InputDict:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: Array, **kwargs) -> InputDict:
        """
        Parameters
        ----------
        inputs:
            Float[Array, "batch time n"]
        kwargs

        """
        input_static_idx, input_known_real_idx, input_known_categorical_idx, input_observed_idx = (
            self.input_static_idx,
            self.input_known_real_idx,
            self.input_known_categorical_idx,
            self.input_observed_idx,
        )

        if input_static_idx is None:
            raise ValueError("When providing inputs as arrays, must specify provide `input_static_idx`")

        if input_known_real_idx is None:
            raise ValueError("When providing inputs as arrays, must specify provide `input_known_real_idx`")

        if input_known_categorical_idx is None:
            raise ValueError("When providing inputs as arrays, must specify provide `input_known_categorical_idx`")

        if input_observed_idx is None:
            raise ValueError("When providing inputs as arrays, must specify provide `input_observed_idx`")

        input_static_idx = ops.array(input_static_idx, "int32")
        input_known_real_idx = ops.array(input_known_real_idx, "int32")
        input_known_categorical_idx = ops.array(input_known_categorical_idx, "int32")
        input_observed_idx = ops.array(input_observed_idx, "int32")

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
                log.error(
                    f"Declared number of features does not match with the one seen in input, "
                    f"could not indentify inputs at {unknown_indexes}"
                )
            else:
                log.error(
                    f"Declared number of features does not match with the one seen in input, "
                    f"no inputs at {unknown_indexes}"
                )

        static = ops.cast(inputs[..., input_static_idx], "int32")

        if len(input_known_real_idx) > 0:
            known_real = ops.cast(inputs[..., input_known_real_idx], self.dtype)
        else:
            known_real = ops.array([])

        if len(input_known_categorical_idx) > 0:
            known_categorical = ops.cast(inputs[..., input_known_categorical_idx], "int32")
        else:
            known_categorical = ops.array([])

        if len(input_observed_idx) > 0:
            observed = ops.cast(inputs[..., input_observed_idx], self.dtype)
        else:
            observed = ops.array([])

        return {
            "static": static,
            "known_real": known_real,
            "observed": observed,
            "known_categorical": known_categorical,
        }


# -------------------------------------------------------------------------------------------------------------


class GatedLinearUnit(layers.Layer):
    def __init__(
        self,
        *,
        latent_dim,
        dropout_rate,
        time_distributed,
        name="GLU",
        **kwargs,
    ):
        """

        Parameters
        ----------
        latent_dim:
            Latent space dimensionality.
        dropout_rate:
            Passed down to layers.Dropout(rate=dropout_rate)
        time_distributed:
            Apply across time axis yes/no.
        name
        kwargs
        """
        super().__init__(name=name, **kwargs)
        self.dropout = layers.Dropout(rate=dropout_rate)
        dense = layers.Dense(latent_dim)
        activation = layers.Dense(latent_dim, activation="sigmoid")

        if time_distributed:
            dense = layers.TimeDistributed(dense)
            activation = layers.TimeDistributed(activation)

        self.dense = dense
        self.activation = activation

    def __call__(self, inputs: Array) -> Tuple[Array, Array]:
        return super().__call__(inputs)

    def call(self, inputs):
        x = self.dropout(inputs)
        x_pre_activation = self.dense(x)
        x_gated = self.activation(x)
        x = x_pre_activation * x_gated
        return x, x_gated


# -------------------------------------------------------------------------------------------------------------
# | resize | time distributed | context |
# |    +   |       -          |    -    |
# |    -   |       -          |    -    |
# |    +   |       +          |    +    |
# |    -   |       +          |    -    |
# |    -   |       +          |    +    |


class GatedResidualNetwork(layers.Layer):
    def __init__(
        self,
        *,
        latent_dim,
        dropout_rate,
        time_distributed,
        output_size=None,
        name="GRU",
        **kwargs,
    ):
        """

        Parameters
        ----------
        latent_dim:
            Latent space dimensionality.
        dropout_rate:
            Dropout rate passed down to `GatedLinearUnit`.
        time_distributed:
            Apply across time axis yes/no.
        output_size:
            Size of output layer, default=latent_dim.
        name
        kwargs

        References
        ----------

        - Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
            - https://arxiv.org/pdf/1912.09363.pdf
            - https://github.com/google-research/google-research/tree/master/tft
        """
        super().__init__(name=name, **kwargs)

        if output_size is None:
            skip_connection = layers.Identity()
        else:
            skip_connection = layers.Dense(output_size)
            if time_distributed:
                skip_connection = layers.TimeDistributed(skip_connection)

        activation = layers.Dense(latent_dim)
        dense = layers.Dense(latent_dim)

        if time_distributed:
            activation = layers.TimeDistributed(activation)
            dense = layers.TimeDistributed(dense)

        self.skip_connection = skip_connection
        self.pre_activation = activation
        self.dense = dense
        self.glu = GatedLinearUnit(
            latent_dim=output_size or latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
        )
        self.ln = layers.LayerNormalization()

    def __call__(self, inputs: Array, **kwargs) -> tuple[Array, Array]:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: Array, **kwargs):
        x_skip = self.skip_connection(inputs)
        x = self.pre_activation(inputs)
        x = ops.nn.elu(x)
        x = self.dense(x)
        x, gate = self.glu(x)
        x = self.ln(x + x_skip)
        return x, gate


class ContextualGatedResidualNetwork(GatedResidualNetwork):
    def __init__(
        self,
        *,
        latent_dim,
        dropout_rate,
        time_distributed,
        output_size=None,
        name="GRU",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
            output_size=output_size,
            name=name,
            **kwargs,
        )

        context_dense = layers.Dense(latent_dim, use_bias=False)
        if time_distributed:
            context_dense = layers.TimeDistributed(context_dense)
        self.context_dense = context_dense

    def __call__(self, inputs: ContextInput, **kwargs) -> tuple[Array, Array]:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: ContextInput, **kwargs):
        x_inputs = inputs["inputs"]
        x_skip = self.skip_connection(x_inputs)
        x = self.pre_activation(x_inputs)
        context = self.context_dense(inputs["context"])
        x += context

        x = ops.nn.elu(x)
        x = self.dense(x)
        x, gate = self.glu(x)
        x = self.ln(x + x_skip)
        return x, gate


# -------------------------------------------------------------------------------------------------------------


class InputEmbedding(layers.Layer):
    def __init__(
        self,
        *,
        static_categories_sizes,
        known_categories_sizes,
        num_known_real_inputs,
        num_observed_inputs,
        latent_dim,
        name="embed",
        **kwargs,
    ):
        """

        This layer project all different inputs into the same latent space.
        Embedding is applied to categorical ones, and real-values ones are
        linearly mapped over `num_time_steps` axis.

        Parameters
        ----------
        static_categories_sizes:
            Sequence of ints describing with max value for each category of static inputs.
            E.g., you have `shop_name` (which can only be "a", "b" or "c") and `location_city`
            (which can only be "A", "B", "C" or "D") as your static inputs, then static_categories_size=[3, 4].
        known_categories_sizes:
            Sequence of ints describing with max value for each category of known categorical inputs.
            This follows the same principle as `static_categories_size`, with only difference that known inputs can
            change overtime. An example of know categorical input can be day of the week.
        num_known_real_inputs
        num_observed_inputs
        latent_dim:
            Latent space dimensionality.
        num_unknown_inputs
        name
        kwargs
        """
        super().__init__(name=name, **kwargs)

        self.static_embeds = [layers.Embedding(size, latent_dim) for size in static_categories_sizes]
        self.known_embeds = [layers.Embedding(size, latent_dim) for size in known_categories_sizes]
        self.known_dense = [layers.TimeDistributed(layers.Dense(latent_dim)) for _ in range(num_known_real_inputs)]
        self.observed_dense = [layers.TimeDistributed(layers.Dense(latent_dim)) for _ in range(num_observed_inputs)]

    def __call__(self, inputs: InputDict, **kwargs) -> EmbeddingDict:
        return super().__call__(inputs, **kwargs)

    def build(self, input_shape: InputDictShape):
        for layer in self.static_embeds:
            layer.build((input_shape["static"][0], input_shape["static"][2]))

        for layer in self.known_embeds:
            layer.build(input_shape["known_categorical"][:2])

        for layer in self.known_dense:
            layer.build((*input_shape["known_real"][:2], 1))

        for layer in self.observed_dense:
            layer.build((*input_shape["observed"][:2], 1))

        self.built = True

    def call(self, inputs: InputDict, **kwargs) -> EmbeddingDict:
        """
        - categorical -> Embedding
        - real -> TimeDistributed(Dense)

        """
        static_input_embeddings = [layer(inputs["static"][:, 0, i]) for i, layer in enumerate_v2(self.known_embeds)]

        static_input_embeddings = ops.stack(static_input_embeddings, axis=1)

        known_categorical_inputs_embeddings = [
            layer(inputs["known_categorical"][..., i]) for i, layer in enumerate_v2(self.known_embeds)
        ]

        known_real_inputs_embeddings = [
            layer(inputs["known_real"][..., i, newaxis]) for i, layer in enumerate_v2(self.known_dense)
        ]

        known_inputs_embeddings = ops.concatenate(
            [
                ops.stack(known_real_inputs_embeddings, axis=-1),
                ops.stack(known_categorical_inputs_embeddings, axis=-1),
            ],
            axis=-1,
        )

        observed_input_embeddings = [
            layer(inputs["observed"][..., i, newaxis]) for i, layer in enumerate_v2(self.observed_dense)
        ]

        observed_input_embeddings = (
            ops.stack(observed_input_embeddings, axis=-1) if len(observed_input_embeddings) > 0 else ops.array([])
        )

        return {
            "static": static_input_embeddings,
            "observed": observed_input_embeddings,
            "known": known_inputs_embeddings,
        }


# -------------------------------------------------------------------------------------------------------------


class StaticCovariatesEncoder(layers.Layer):
    def __init__(
        self,
        *,
        latent_dim,
        dropout_rate,
        num_static_inputs,
        name="static_encoder",
        **kwargs,
    ):
        """
        Create a static context out of static input embeddings.
            Static context is a (enrichment) vector, which must be added to other time varying inputs during
            `variable selection` process. Additionally, this layer creates initial state for LSTM cells,
            this way we give them `pre-memory`, and model the fact that static inputs do influence time dependent ones.

        Parameters
        ----------
        latent_dim:
        Dimensionality of the latent space.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        num_static_inputs
        name
        kwargs
        """
        super().__init__(name=name, **kwargs)

        def make_gru(**kw):
            return GatedResidualNetwork(latent_dim=latent_dim, dropout_rate=dropout_rate, time_distributed=False, **kw)

        self.gru = make_gru(
            output_size=num_static_inputs,
        )
        self.gru_blocks = [make_gru() for _ in range(num_static_inputs)]
        self.gru_1 = make_gru()
        self.gru_2 = make_gru()
        self.gru_3 = make_gru()
        self.gru_4 = make_gru()
        self.flatten = layers.Flatten()

    def __call__(self, inputs: Array, **kwargs) -> StaticContextDict:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: Array, **kwargs) -> StaticContextDict:
        flat_x = self.flatten(inputs)
        mlp_outputs, _ = self.gru(flat_x)

        sparse_weights = ops.nn.softmax(mlp_outputs)[..., newaxis]

        transformed_embeddings = [layer(inputs[:, i, newaxis])[0] for i, layer in enumerate_v2(self.gru_blocks)]

        transformed_embeddings = ops.concatenate(transformed_embeddings, axis=1)
        static_context_vector = ops.sum(sparse_weights * transformed_embeddings, axis=1)
        context_variable_selection, _ = self.gru_1(static_context_vector)
        context_enrichment, _ = self.gru_2(static_context_vector)
        context_state_h, _ = self.gru_3(static_context_vector)
        context_state_c, _ = self.gru_4(static_context_vector)
        return {
            "enrichment": context_enrichment,
            "weight": sparse_weights,
            "state_c": context_state_c,
            "state_h": context_state_h,
            "vector": context_variable_selection,
        }


# -------------------------------------------------------------------------------------------------------------


class VariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        *,
        latent_dim,
        dropout_rate,
        num_time_steps,
        num_inputs,
        name="variable_selection",
        **kwargs,
    ):
        """

        Parameters
        ----------
        latent_dim:
            Latent space dimensionality.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        num_time_steps:
            Size of 2nd axis in the input.
        num_inputs:
            Size of 3rd axis in the input.
        name
        kwargs
        """
        super().__init__(name=name, **kwargs)
        self.num_time_steps = num_time_steps
        self.latent_dim = latent_dim
        self.num_inputs = num_inputs

        self.context_grn = ContextualGatedResidualNetwork(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            output_size=num_inputs,
            time_distributed=True,
        )
        self.gru = [
            GatedResidualNetwork(
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                time_distributed=True,
            )
            for _ in range(num_inputs)
        ]

    def __call__(self, inputs: ContextInput, **kwargs) -> tuple[Array, Array, Array]:
        return super().__call__(inputs, **kwargs)

    def call(self, inputs: ContextInput, **kwargs):
        mlp_outputs, static_gate = self.context_grn(
            {
                "inputs": ops.reshape(inputs["inputs"], [-1, self.num_time_steps, self.latent_dim * self.num_inputs]),
                "context": ops.expand_dims(inputs["context"], axis=1),
            }
        )

        sparse_weights = ops.nn.softmax(mlp_outputs)[:, :, newaxis]

        transformed_embeddings = [layer(inputs["inputs"][..., i])[0] for i, layer in enumerate_v2(self.gru)]

        transformed_embeddings = ops.stack(transformed_embeddings, axis=-1)
        temporal_ctx = ops.sum(sparse_weights * transformed_embeddings, axis=-1)
        return temporal_ctx, sparse_weights, static_gate


class DecoderBlock(layers.Layer):
    def __init__(
        self,
        num_attention_heads,
        latent_dim,
        dropout_rate,
        attention_dropout_rate=0.1,
        name="decoder",
        **kwargs,
    ):
        """

        Parameters
        ----------
        num_attention_heads:
            Number of attention heads for MultiHeadSelfAttention.
        latent_dim:
            Latent space dimensionality.
        dropout_rate:
            Dropout rate passed down to keras.layer.Dropout.
        attention_dropout_rate
        name
        kwargs
        """
        super().__init__(name=name, **kwargs)
        self.sa = layers.MultiHeadAttention(
            num_heads=num_attention_heads, dropout=attention_dropout_rate, key_dim=latent_dim, use_bias=False
        )
        self.glu = GatedLinearUnit(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=True,
        )
        self.ln = layers.LayerNormalization()
        self.gru = GatedResidualNetwork(
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            time_distributed=True,
        )

    def call(self, inputs: Array, **kwargs) -> Array:
        x = self.sa(inputs, inputs, use_causal_mask=True)
        x, _ = self.glu(x)
        x = self.ln(x + inputs)
        # Nonlinear processing on outputs
        decoded, _ = self.gru(x)
        return decoded


# ------------------------------------------------------------------------------------


class ContextInput(TypedDict):
    inputs: Array
    context: Array


class EmbeddingDict(TypedDict):
    """
    Attributes
    ----------

    static:
        3D float
    known:
        3D float
    observed:
        3D float
    """

    static: Array
    known: Array
    observed: Array


class StaticContextDict(TypedDict):
    """
    Attributes
    ----------

    enrichment:
        has shape (batch_size, latent_dim) and must be used as additional context for GRNs.
    vector:
        has shape (batch_size, latent_dim), must be passed down to temporal decoder, and used  as additional context in GRNs.
    state_c:
        has shape (batch_size, latent_dim) and must be used together with `state_h` as initial context for LSTM cells.
    state_h
        has shape (batch_size, latent_dim) and must be used together with `state_c` as initial context for LSTM cells.
    """

    enrichment: Array
    state_h: Array
    state_c: Array
    vector: Array
    weight: Array


class InputDict(TypedDict):
    """
    Attributes
    ----------
    static:
        3D int
    known_real:
        3D float
    known_categorical:
        3D int
    observed:
        3D float
    """

    static: Array
    known_real: Array
    known_categorical: Array
    observed: Array


class InputDictShape(TypedDict):
    static: Tuple[int, ...] | Tuple[int, int, int]
    known_real: Tuple[int, ...] | Tuple[int, int, int]
    known_categorical: Tuple[int, ...] | Tuple[int, int, int]
    observed: Tuple[int, ...] | Tuple[int, int, int]


class TftOutputs(TypedDict):
    logits: Array
    static_flags: Array
    historical_flags: Array
    future_flags: Array
