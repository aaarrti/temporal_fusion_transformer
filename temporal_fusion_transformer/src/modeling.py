from __future__ import annotations

import logging
from typing import Sequence, Tuple

import keras_core as keras
from keras_core import KerasTensor, layers, ops, backend
from keras_core.mixed_precision import global_policy
from toolz import functoolz

from temporal_fusion_transformer.src.utils import enumerate_v2, zip_v2

newaxis = None

log = logging.getLogger(__name__)


def TemporalFusionTransformer(
    *,
    input_observed_idx: Sequence[int],
    input_static_idx: Sequence[int],
    input_known_real_idx: Sequence[int],
    input_known_categorical_idx: Sequence[int],
    static_categories_sizes: Sequence[int],
    known_categories_sizes: Sequence[int],
    hidden_layer_size: int,
    dropout_rate: float,
    encoder_steps: int,
    total_time_steps: int,
    num_attention_heads: int,
    num_decoder_blocks: int,
    num_quantiles: int,
    num_outputs: int = 1,
    return_attention: bool = False,
    unroll: bool = False,
    **kwargs,
):
    """
    References
    ----------

    Bryan Lim, et.al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/pdf/1912.09363.pdf, https://github.com/google-research/google-research/tree/master/tft

    Parameters
    ----------
    input_observed_idx
    input_static_idx
    input_known_real_idx
    input_known_categorical_idx
    static_categories_sizes
    known_categories_sizes
    hidden_layer_size
    dropout_rate
    encoder_steps
    total_time_steps
    num_attention_heads
    num_decoder_blocks
    num_quantiles
    num_outputs
    return_attention
    unroll
    kwargs

    Returns
    -------

    """

    num_features = sum(
        [
            len(i)
            for i in [
                input_static_idx,
                input_known_real_idx,
                input_observed_idx,
                input_known_categorical_idx,
            ]
        ]
    )

    inputs = keras.Input(
        shape=(total_time_steps, num_features),
        name="inputs",
        dtype=global_policy().compute_dtype,
    )

    static_inputs, known_combined_layer, obs_inputs = apply_tft_embeddings(
        inputs,
        input_observed_idx=input_observed_idx,
        input_static_idx=input_static_idx,
        input_known_real_idx=input_known_real_idx,
        static_categories_sizes=static_categories_sizes,
        known_categories_sizes=known_categories_sizes,
        input_known_categorical_idx=input_known_categorical_idx,
        hidden_layer_size=hidden_layer_size,
    )

    if obs_inputs is not None:
        historical_inputs = ops.concatenate(
            [known_combined_layer[:, :encoder_steps, :], obs_inputs[:, :encoder_steps, :]], axis=-1
        )
    else:
        historical_inputs = known_combined_layer[:, :encoder_steps, :]

    # Isolate only known future inputs.
    future_inputs = known_combined_layer[:, encoder_steps:, :]

    static_encoder, static_weights = static_combine_and_mask(
        static_inputs,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        num_static=len(input_static_idx),
    )

    static_context_variable_selection = gated_residual_network(
        static_encoder,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
    )
    static_context_enrichment = gated_residual_network(
        static_encoder,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
    )
    static_context_state_h = gated_residual_network(
        static_encoder,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
    )
    static_context_state_c = gated_residual_network(
        static_encoder,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
    )

    historical_features, historical_flags, _ = lstm_combine_and_mask(
        historical_inputs,
        static_context_variable_selection,
        dropout_rate=dropout_rate,
        hidden_layer_size=hidden_layer_size,
    )
    future_features, future_flags, _ = lstm_combine_and_mask(
        future_inputs,
        static_context_variable_selection,
        dropout_rate=dropout_rate,
        hidden_layer_size=hidden_layer_size,
    )

    history_lstm, state_h, state_c = layers.LSTM(
        hidden_layer_size, return_sequences=True, return_state=True, unroll=unroll
    )(historical_features, initial_state=[static_context_state_h, static_context_state_c])

    future_lstm = layers.LSTM(
        hidden_layer_size, return_sequences=True, return_state=False, unroll=unroll
    )(future_features, initial_state=[state_h, state_c])

    lstm_layer = ops.concatenate([history_lstm, future_lstm], axis=1)

    # Apply gated skip connection
    input_embeddings = ops.concatenate([historical_features, future_features], axis=1)

    lstm_layer, _ = apply_gating_layer(
        lstm_layer, hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate, activation=None
    )
    temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

    # Static enrichment layers
    expanded_static_context = ops.expand_dims(static_context_enrichment, axis=1)
    enriched, _ = gated_residual_network(
        temporal_feature_layer,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True,
    )

    transformer_layer = enriched

    for _ in range(num_decoder_blocks):
        transformer_layer = apply_transformer_layer(
            transformer_layer,
            num_attention_heads=num_attention_heads,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
        )
        transformer_layer = add_and_norm([transformer_layer, temporal_feature_layer])

    outputs = ops.stack(
        [
            layers.TimeDistributed(layers.Dense(num_quantiles))(
                transformer_layer[..., encoder_steps:, i, newaxis]
            )
            for i in range(num_outputs)
        ],
        axis=-2,
    )

    return keras.Model(inputs, outputs, **kwargs)


# -------------------------------------------------------------------------------------------------------------


def apply_tft_embeddings(
    inputs: keras.KerasTensor,
    *,
    static_categories_sizes: Sequence[int],
    known_categories_sizes: Sequence[int],
    input_observed_idx: Sequence[int],
    input_static_idx: Sequence[int],
    input_known_real_idx: Sequence[int],
    input_known_categorical_idx: Sequence[int],
    hidden_layer_size: int,
) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor | None]:
    def convert_real_to_embedding(x, name=None):
        """Applies linear transformation for time-varying inputs."""
        return layers.TimeDistributed(layers.Dense(hidden_layer_size), name=name)(x)

    static_embeddings = ops.stack(
        [
            layers.Embedding(size, hidden_layer_size)(inputs[..., 0, idx])
            for idx, size in zip_v2(input_static_idx, static_categories_sizes)
        ],
        axis=1,
    )

    known_categorical_inputs_embeddings = [
        layers.Embedding(size, hidden_layer_size)(inputs[..., idx])
        for idx, size in zip_v2(input_known_categorical_idx, known_categories_sizes)
    ]
    known_real_inputs_embeddings = [
        convert_real_to_embedding(inputs[..., idx, newaxis])
        for i, idx in enumerate_v2(input_known_real_idx)
    ]

    known_inputs_embeddings = ops.concatenate(
        [
            ops.stack(known_real_inputs_embeddings, axis=-1),
            ops.stack(known_categorical_inputs_embeddings, axis=-1),
        ],
        axis=-1,
    )

    if len(input_observed_idx) != 0:
        observed_embeddings = ops.stack(
            [convert_real_to_embedding(ops.take(inputs, [i], axis=-1)) for i in input_observed_idx],
            axis=-1,
        )
    else:
        observed_embeddings = None

    return static_embeddings, known_inputs_embeddings, observed_embeddings


def apply_transformer_layer(
    inputs: KerasTensor,
    *,
    num_attention_heads: int,
    hidden_layer_size: int,
    dropout_rate: float,
) -> KerasTensor:
    x = layers.MultiHeadAttention(
        num_attention_heads,
        hidden_layer_size,
        hidden_layer_size,
        dropout=dropout_rate,
        use_bias=False,
    )(inputs, inputs, use_causal_mask=True)

    x, _ = apply_gating_layer(
        x, hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate, activation=None
    )
    x = add_and_norm([x, inputs])

    # Nonlinear processing on outputs
    decoder = gated_residual_network(
        x, hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate, use_time_distributed=True
    )

    # Final skip connection
    decoder, _ = apply_gating_layer(decoder, hidden_layer_size=hidden_layer_size, activation=None)

    return decoder


# -------------------------------------------------------------------------------------------------------------


def linear_layer(
    size: int,
    activation: str | None = None,
    use_time_distributed: bool = False,
    use_bias: bool = True,
) -> layers.Layer:
    """Returns simple Keras linear layer.

    Args:
      size: Output size
      activation: Activation function to apply if required
      use_time_distributed: Whether to apply layer across time
      use_bias: Whether bias should be included in layer
    """
    linear = layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = layers.TimeDistributed(linear)
    return linear


def apply_gating_layer(
    x: KerasTensor,
    hidden_layer_size: int,
    dropout_rate: float | None = None,
    use_time_distributed: bool = True,
    activation: str | None = None,
) -> Tuple[KerasTensor, KerasTensor]:
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
      x: Input to gating layer
      hidden_layer_size: Dimension of GLU
      dropout_rate: Dropout rate to apply if any
      use_time_distributed: Whether to apply across time
      activation: Activation function to apply to the linear feature transform if
        necessary

    Returns:
      Tuple of tensors for: (GLU output, gate)
    """

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = layers.TimeDistributed(
            layers.Dense(hidden_layer_size, activation=activation)
        )(x)
        gated_layer = layers.TimeDistributed(layers.Dense(hidden_layer_size, activation="sigmoid"))(
            x
        )
    else:
        activation_layer = layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = layers.Dense(hidden_layer_size, activation="sigmoid")(x)

    return layers.Multiply()([activation_layer, gated_layer]), gated_layer


def add_and_norm(x_Sequence: Sequence[KerasTensor]) -> KerasTensor:
    """Applies skip connection followed by layer normalisation.

    Args:
      x_Sequence: Sequence of inputs to sum for skip connection

    Returns:
      Tensor output from layer.
    """
    tmp = layers.Add()(x_Sequence)
    tmp = layers.LayerNormalization()(tmp)
    return tmp


def gated_residual_network(
    x: KerasTensor,
    *,
    hidden_layer_size: int,
    output_size: int | None = None,
    dropout_rate: float = None,
    use_time_distributed: bool = True,
    additional_context: keras.KerasTensor | None = None,
    return_gate: bool = False,
) -> Tuple[KerasTensor, KerasTensor] | KerasTensor:
    """Applies the gated residual network (GRN) as defined in paper.

    Args:
      x: Network inputs
      hidden_layer_size: Internal state size
      output_size: Size of output layer
      dropout_rate: Dropout rate if dropout is applied
      use_time_distributed: Whether to apply network across time dimension
      additional_context: Additional context vector to use if relevant
      return_gate: Whether to return GLU gate for diagnostic purposes

    Returns:
      Tuple of tensors for: (GRN output, GLU gate)
    """

    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = layers.Dense(output_size)
        if use_time_distributed:
            linear = layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False,
        )(additional_context)
    hidden = layers.Activation("elu")(hidden)
    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None,
    )

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


def static_combine_and_mask(
    embedding: KerasTensor, *, num_static: int, hidden_layer_size: int, dropout_rate: float
) -> Tuple[KerasTensor, KerasTensor]:
    """Applies variable selection network to static inputs.

    Args:
      embedding: Transformed static inputs
      hidden_layer_size
      dropout_rate
      num_static

    Returns:
      Tensor output for variable selection network
    """

    # Add temporal features

    flatten = layers.Flatten()(embedding)

    # Nonlinear transformation with gated residual network.
    mlp_outputs = gated_residual_network(
        flatten,
        hidden_layer_size=hidden_layer_size,
        output_size=num_static,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
        additional_context=None,
    )

    sparse_weights = ops.nn.softmax(mlp_outputs)
    sparse_weights = ops.expand_dims(sparse_weights, axis=-1)

    trans_emb_Sequence = []
    for i in range(num_static):
        e = gated_residual_network(
            embedding[:, i : i + 1, :],
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
        )
        trans_emb_Sequence.append(e)

    transformed_embedding = ops.concatenate(trans_emb_Sequence, axis=1)

    combined = layers.Multiply()([sparse_weights, transformed_embedding])

    static_vec = ops.sum(combined, axis=1)

    return static_vec, sparse_weights


def lstm_combine_and_mask(
    embedding: KerasTensor,
    static_context_variable_selection: KerasTensor,
    *,
    dropout_rate: float,
    hidden_layer_size: int,
) -> Tuple[KerasTensor, KerasTensor, KerasTensor]:
    """Apply temporal variable selection networks.

    Args:
      embedding: Transformed inputs.
      static_context_variable_selection

      dropout_rate
      hidden_layer_size

    Returns:
      Processed tensor outputs.
    """

    # Add temporal features
    _, time_steps, embedding_dim, num_inputs = ops.shape(embedding)

    flatten = ops.reshape(embedding, [-1, time_steps, embedding_dim * num_inputs])

    expanded_static_context = ops.expand_dims(static_context_variable_selection, axis=1)

    # Variable selection weights
    mlp_outputs, static_gate = gated_residual_network(
        flatten,
        hidden_layer_size=hidden_layer_size,
        output_size=num_inputs,
        dropout_rate=dropout_rate,
        use_time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True,
    )

    sparse_weights = layers.Activation("softmax")(mlp_outputs)
    sparse_weights = ops.expand_dims(sparse_weights, axis=2)

    # Non-linear Processing & weight application
    trans_emb_Sequence = []
    for i in range(num_inputs):
        grn_output = gated_residual_network(
            embedding[..., i],
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
        )
        trans_emb_Sequence.append(grn_output)

    transformed_embedding = ops.stack(trans_emb_Sequence, axis=-1)

    combined = layers.Multiply()([sparse_weights, transformed_embedding])
    temporal_ctx = ops.sum(combined, axis=-1)

    return temporal_ctx, sparse_weights, static_gate
