from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

import keras
import numpy as np
from keras import layers, ops

from temporal_fusion_transformer.src.config import Config, RegularizerT
from temporal_fusion_transformer.src.modeling.tft_layers import (
    AddAndNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    GatedResidualNetworkWithContext,
    InputEmbedding,
    Linear,
    StaticVariableSelectionNetwork,
    TransformerBlock,
    VariableSelectionNetwork,
)
from temporal_fusion_transformer.src.utils import count_inputs

newaxis = None

if TYPE_CHECKING:
    import tensorflow as tf


class ContextInput(TypedDict):
    input: tf.Tensor
    context: tf.Tensor


class TemporalFusionTransformer(keras.Model):
    def __init__(
        self,
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
        num_attention_heads: int,
        num_decoder_blocks: int,
        num_quantiles: int,
        num_outputs: int,
        total_time_steps: int,
        kernel_regularizer: RegularizerT,
        bias_regularizer: RegularizerT,
        activity_regularizer: RegularizerT,
        recurrent_regularizer: RegularizerT,
        embeddings_regularizer: RegularizerT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_steps = encoder_steps
        self.total_time_steps = total_time_steps
        self.num_decoder_blocks = num_decoder_blocks
        self.num_quantiles = num_quantiles
        self.num_outputs = num_outputs

        num_static = len(input_static_idx)
        num_non_static = (
            len(input_observed_idx) + len(input_known_categorical_idx) + len(input_known_real_idx)
        )

        self.embedding = InputEmbedding(
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=known_categories_sizes,
            input_observed_idx=input_observed_idx,
            input_static_idx=input_static_idx,
            input_known_real_idx=input_known_real_idx,
            input_known_categorical_idx=input_known_categorical_idx,
            hidden_layer_size=hidden_layer_size,
            embeddings_regularizer=embeddings_regularizer,
        )

        self.static_combine_and_mask = StaticVariableSelectionNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            num_static=num_static,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.static_context_variable_selection = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.static_context_enrichment = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.static_context_state_h = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.static_context_state_c = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.historical_variable_selection = VariableSelectionNetwork(
            num_inputs=num_non_static,
            dropout_rate=dropout_rate,
            hidden_layer_size=hidden_layer_size,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.historical_lstm = layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            return_state=True,
            # We need unroll to prevent None shape
            unroll=True,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            recurrent_regularizer=recurrent_regularizer,
        )

        self.future_variable_selections = VariableSelectionNetwork(
            num_inputs=num_non_static,
            dropout_rate=dropout_rate,
            hidden_layer_size=hidden_layer_size,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.future_lstm = layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            return_state=False,
            # We need unroll to prevent None shape
            unroll=True,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            recurrent_regularizer=recurrent_regularizer,
        )

        self.lstm_gate = GatedLinearUnit(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.lstm_add_norm = AddAndNorm()

        self.enriched_grn = GatedResidualNetworkWithContext(
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.transformer_blocks = [
            TransformerBlock(
                num_attention_heads=num_attention_heads,
                dropout_rate=dropout_rate,
                hidden_layer_size=hidden_layer_size,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
            )
            for _ in range(num_decoder_blocks)
        ]
        self.transformer_add_norm = [AddAndNorm() for _ in range(num_decoder_blocks)]
        self.output_projection = Linear(
            num_outputs * num_quantiles,
            use_time_distributed=True,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
        )

    def call(self, inputs, training=False):
        static_inputs, known_combined_layer, obs_inputs = self.embedding(inputs)

        encoder_steps = self.encoder_steps

        if obs_inputs is not None:
            historical_inputs = ops.concatenate(
                [known_combined_layer[:, :encoder_steps, :], obs_inputs[:, :encoder_steps, :]],
                axis=-1,
            )
        else:
            historical_inputs = known_combined_layer[:, :encoder_steps, :]

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)

        static_context_variable_selection, _ = self.static_context_variable_selection(
            static_encoder
        )
        static_context_enrichment, _ = self.static_context_enrichment(
            static_encoder,
        )
        static_context_state_h, _ = self.static_context_state_h(
            static_encoder,
        )
        static_context_state_c, _ = self.static_context_state_c(
            static_encoder,
        )

        historical_features, historical_flags, _ = self.historical_variable_selection(
            {
                "input": historical_inputs,
                "context": static_context_variable_selection,
            }
        )

        history_lstm, state_h, state_c = self.historical_lstm(
            historical_features, initial_state=[static_context_state_h, static_context_state_c]
        )

        future_features, future_flags, _ = self.future_variable_selections(
            {"input": future_inputs, "context": static_context_variable_selection},
        )

        future_lstm = self.future_lstm(future_features, initial_state=[state_h, state_c])

        lstm_layer = ops.concatenate([history_lstm, future_lstm], axis=1)
        input_embeddings = ops.concatenate([historical_features, future_features], axis=1)

        lstm_layer, _ = self.lstm_gate(lstm_layer)

        temporal_feature_layer = self.lstm_add_norm([lstm_layer, input_embeddings])

        # Static enrichment layers
        expanded_static_context = ops.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = self.enriched_grn(
            {"input": temporal_feature_layer, "context": expanded_static_context}
        )

        transformer_layer = enriched

        for i in range(self.num_decoder_blocks):
            transformer_layer = self.transformer_blocks[i](transformer_layer)
            transformer_layer = self.transformer_add_norm[i](
                [transformer_layer, temporal_feature_layer]
            )

        outputs = self.output_projection(transformer_layer[:, encoder_steps:])
        outputs = ops.reshape(
            outputs,
            (
                -1,
                self.total_time_steps - encoder_steps,
                self.num_outputs,
                self.num_quantiles,
            ),
        )
        return outputs

    @staticmethod
    def from_dataclass_config(config: Config) -> TemporalFusionTransformer:
        return TemporalFusionTransformer(
            input_observed_idx=config.input_observed_idx,
            input_static_idx=config.input_static_idx,
            input_known_real_idx=config.input_known_real_idx,
            input_known_categorical_idx=config.input_known_categorical_idx,
            static_categories_sizes=config.static_categories_sizes,
            known_categories_sizes=config.known_categories_sizes,
            hidden_layer_size=config.hidden_layer_size,
            dropout_rate=config.dropout_rate,
            encoder_steps=config.encoder_steps,
            num_attention_heads=config.num_attention_heads,
            num_decoder_blocks=config.num_decoder_blocks,
            num_outputs=config.num_outputs,
            num_quantiles=len(config.quantiles),
            total_time_steps=config.total_time_steps,
            activity_regularizer=config.activity_regularizer,
            kernel_regularizer=config.kernel_regularizer,
            recurrent_regularizer=config.recurrent_regularizer,
            bias_regularizer=config.bias_regularizer,
            embeddings_regularizer=config.embeddings_regularizer,
        )

    @staticmethod
    def build_from_dataclass_config(
        config: Config, weights_path: str | None = None
    ) -> TemporalFusionTransformer:
        model = TemporalFusionTransformer.from_dataclass_config(config)
        num_inputs = count_inputs(config)

        x = np.ones(shape=(1, config.total_time_steps, num_inputs), dtype=np.float32)
        model.predict(x, verbose=False)
        if weights_path is not None:
            model.load_weights(weights_path)
        return model

    def compile(
        self,
        optimizer="adam",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile=False,
        auto_scale_loss=True,
    ):
        # just to overwrite the default for `jit_compile` and optimizer
        super().compile(
            optimizer,
            loss,
            loss_weights,
            metrics,
            weighted_metrics,
            steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
        )
