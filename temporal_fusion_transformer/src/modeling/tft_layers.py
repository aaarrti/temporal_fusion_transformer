from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

from keras import layers, ops

newaxis = None

if TYPE_CHECKING:
    import tensorflow as tf


class ContextInput(TypedDict):
    input: tf.Tensor
    context: tf.Tensor


class InputEmbedding(layers.Layer):
    def __init__(
        self,
        *,
        static_categories_sizes: Sequence[int],
        known_categories_sizes: Sequence[int],
        input_observed_idx: Sequence[int],
        input_static_idx: Sequence[int],
        input_known_real_idx: Sequence[int],
        input_known_categorical_idx: Sequence[int],
        hidden_layer_size: int,
        embeddings_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.static_embeddings = [
            layers.Embedding(size, hidden_layer_size, embeddings_regularizer=embeddings_regularizer)
            for size in static_categories_sizes
        ]

        self.known_categorical_inputs_embeddings = [
            layers.Embedding(size, hidden_layer_size, embeddings_regularizer=embeddings_regularizer)
            for size in known_categories_sizes
        ]

        self.input_static_idx = input_static_idx
        self.input_known_categorical_idx = input_known_categorical_idx

        self.known_real_projection = [
            Linear(
                hidden_layer_size,
                use_time_distributed=True,
                kernel_regularizer=embeddings_regularizer,
            )
            for _ in input_known_real_idx
        ]

        self.observed_projection = [
            Linear(
                hidden_layer_size,
                use_time_distributed=True,
                kernel_regularizer=embeddings_regularizer,
            )
            for _ in input_observed_idx
        ]

        self.input_known_real_idx = list(input_known_real_idx)
        self.input_observed_idx = list(input_observed_idx)

    def call(self, inputs, *args, **kwargs):
        static_embeddings = ops.stack(
            [
                layer(inputs[..., 0, idx])
                for idx, layer in zip(self.input_static_idx, self.static_embeddings)
            ],
            axis=1,
        )
        known_categorical_inputs_embeddings = [
            layer(inputs[..., idx])
            for idx, layer in zip(
                self.input_known_categorical_idx, self.known_categorical_inputs_embeddings
            )
        ]

        known_real_inputs_embeddings = [
            layer(inputs[..., i, newaxis])
            for i, layer in zip(self.input_known_real_idx, self.known_real_projection)
        ]

        known_inputs_embeddings = ops.concatenate(
            [
                ops.stack(known_real_inputs_embeddings, axis=-1),
                ops.stack(known_categorical_inputs_embeddings, axis=-1),
            ],
            axis=-1,
        )

        if len(self.input_observed_idx) != 0:
            observed_embeddings = ops.stack(
                [
                    layer(inputs[..., i, newaxis])
                    for i, layer in zip(self.input_observed_idx, self.observed_projection)
                ],
                axis=-1,
            )
        else:
            observed_embeddings = None

        return static_embeddings, known_inputs_embeddings, observed_embeddings


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        *,
        num_attention_heads: int,
        hidden_layer_size: int,
        dropout_rate: float,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_layer_size,
            value_dim=hidden_layer_size,
            dropout=dropout_rate,
            use_bias=False,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )
        self.glu_1 = GatedLinearUnit(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            activation=None,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )
        self.add_and_norm = AddAndNorm()
        self.grn = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )

        self.glu_2 = GatedLinearUnit(
            hidden_layer_size=hidden_layer_size,
            activation=None,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, inputs, **kwargs):
        x = self.mha(inputs, inputs, use_causal_mask=True)
        x, _ = self.glu_1(x)
        x = self.add_and_norm([x, inputs])
        # Nonlinear processing on outputs
        decoder, _ = self.grn(x)
        # Final skip connection
        decoder, _ = self.glu_2(decoder)
        return decoder


# -------------------------------------------------------------------------------------------------------------


class Linear(layers.Layer):
    def __init__(
        self,
        size: int,
        *,
        activation: str | None = None,
        use_time_distributed: bool = False,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        linear = layers.Dense(
            size,
            activation=activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )
        if use_time_distributed:
            linear = layers.TimeDistributed(linear)

        self.linear = linear

    def call(self, inputs, **kwargs):
        return self.linear(inputs)


class GatedLinearUnit(layers.Layer):
    def __init__(
        self,
        *,
        hidden_layer_size: int,
        dropout_rate: float | None = None,
        use_time_distributed: bool = True,
        activation: str | None = None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = (
            layers.Dropout(dropout_rate) if dropout_rate is not None else layers.Identity()
        )
        self.activation = Linear(
            hidden_layer_size,
            activation=activation,
            use_time_distributed=use_time_distributed,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self.gate = Linear(
            hidden_layer_size,
            activation="sigmoid",
            use_time_distributed=use_time_distributed,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )

    def call(self, x: tf.Tensor, **kwargs):
        x = self.dropout(x)
        activation = self.activation(x)
        gate = self.gate(x)
        return ops.multiply(activation, gate), gate


class AddAndNorm(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def call(self, x, **kwargs):
        x = self.add(x)
        x = self.norm(x)
        return x


class GatedResidualNetwork(layers.Layer):
    def __init__(
        self,
        *,
        hidden_layer_size: int,
        output_size: int | None = None,
        dropout_rate: float | None = None,
        use_time_distributed: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if output_size is None:
            output_size = hidden_layer_size
            skip = layers.Identity()
        else:
            skip = Linear(
                output_size,
                use_time_distributed=use_time_distributed,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
            )

        self.skip = skip
        self.hidden = Linear(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self.elu = layers.Activation("elu")
        self.hidden_2 = Linear(
            hidden_layer_size,
            use_time_distributed=use_time_distributed,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self.glu = GatedLinearUnit(
            hidden_layer_size=output_size,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self.add_and_norm = AddAndNorm()

    def call(self, x: tf.Tensor, context: tf.Tensor | None = None, **kwargs):
        # Setup skip connection
        skip = self.skip(x)
        # Apply feedforward network
        hidden = self.hidden(x)
        hidden = self.elu(hidden)
        hidden = self.hidden_2(hidden)
        gating_layer, gate = self.glu(hidden)
        return self.add_and_norm([skip, gating_layer]), gate


class GatedResidualNetworkWithContext(GatedResidualNetwork):
    def __init__(
        self,
        *,
        hidden_layer_size: int,
        output_size: int | None = None,
        dropout_rate: float | None = None,
        use_time_distributed: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            hidden_layer_size=hidden_layer_size,
            output_size=output_size,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
            **kwargs,
        )
        self.context_linear = Linear(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )

    def call(self, x: ContextInput, **kwargs):
        inputs = x["input"]
        context = x["context"]

        # Setup skip connection
        skip = self.skip(inputs)
        # Apply feedforward network
        hidden = self.hidden(inputs)

        if context is not None:
            hidden = hidden + self.context_linear(context)

        hidden = self.elu(hidden)
        hidden = self.hidden_2(hidden)
        gating_layer, gate = self.glu(hidden)

        return self.add_and_norm([skip, gating_layer]), gate


class StaticVariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        *,
        num_static: int,
        hidden_layer_size: int,
        dropout_rate: float,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.grn = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            output_size=num_static,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self.grn_list = [
            GatedResidualNetwork(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                use_time_distributed=False,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
            )
            for _ in range(num_static)
        ]
        self.num_static = num_static

    def call(self, x: tf.Tensor, **kwargs) -> tuple[tf.Tensor, tf.Tensor]:
        flatten = self.flatten(x)

        # Nonlinear transformation with gated residual network.
        mlp_outputs, _ = self.grn(flatten)

        sparse_weights = ops.nn.softmax(mlp_outputs)
        sparse_weights = ops.expand_dims(sparse_weights, axis=-1)

        trans_emb_Sequence = []
        for i in range(self.num_static):
            e, _ = self.grn_list[i](x[:, i : i + 1, :])
            trans_emb_Sequence.append(e)

        transformed_embedding = ops.concatenate(trans_emb_Sequence, axis=1)

        combined = layers.Multiply()([sparse_weights, transformed_embedding])

        static_vec = ops.sum(combined, axis=1)

        return static_vec, sparse_weights


class VariableSelectionNetwork(layers.Layer):
    def __init__(
        self,
        *,
        dropout_rate: float,
        hidden_layer_size: int,
        num_inputs: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grn = GatedResidualNetworkWithContext(
            hidden_layer_size=hidden_layer_size,
            output_size=num_inputs,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )

        self.grn_list = [
            GatedResidualNetwork(
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                use_time_distributed=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
            )
            for _ in range(num_inputs)
        ]

        self.num_inputs = num_inputs
        self.mul = layers.Multiply()

    def call(self, x: ContextInput, **kwargs):
        embedding = x["input"]
        context = x["context"]

        # Add temporal features
        _, time_steps, embedding_dim, _ = ops.shape(embedding)

        flatten = ops.reshape(embedding, [-1, time_steps, embedding_dim * self.num_inputs])

        # must be (None, 5)
        expanded_static_context = ops.expand_dims(context, axis=1)

        # Variable selection weights
        mlp_outputs, static_gate = self.grn({"input": flatten, "context": expanded_static_context})

        sparse_weights = layers.Activation("softmax")(mlp_outputs)
        sparse_weights = ops.expand_dims(sparse_weights, axis=2)

        # Non-linear Processing & weight application
        trans_emb_Sequence = []
        for i in range(self.num_inputs):
            grn_output, _ = self.grn_list[i](embedding[..., i])
            trans_emb_Sequence.append(grn_output)

        transformed_embedding = ops.stack(trans_emb_Sequence, axis=-1)

        combined = self.mul([sparse_weights, transformed_embedding])
        temporal_ctx = ops.sum(combined, axis=-1)

        return temporal_ctx, sparse_weights, static_gate
