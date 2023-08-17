import pytest

import jax
import numpy as np
import chex

from temporal_fusion_transformer.src.tft_layers import (
    StaticCovariatesEncoder,
    InputEmbedding,
    VariableSelectionNetwork,
    DecoderBlock,
    InputStruct,
    EmbeddingStruct,
    StaticContextStruct,
)

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5
PRNG_KEY = jax.random.PRNGKey(0)


def test_input_embedding():
    x_batch = InputStruct(
        static=np.ones((8, 30, 4), dtype=np.int32),
        known_real=np.ones((8, 30, 3), dtype=np.float32),
        known_categorical=np.ones((8, 30, 2), dtype=np.int32),
        observed=np.random.default_rng(69).uniform(size=(8, 30, 1)),
    )
    layer = InputEmbedding(
        static_categories_sizes=static_categories_sizes,
        known_categories_sizes=known_categories_sizes,
        latent_dim=hidden_layer_size,
        num_observed_inputs=1,
        num_known_real_inputs=3,
    )
    embeds, _ = layer.init_with_output(PRNG_KEY, x_batch)
    embeds: EmbeddingStruct
    # Same shapes as in the original implementation.
    chex.assert_shape(embeds.static, (batch_size, 2, hidden_layer_size))
    chex.assert_shape(embeds.known, (batch_size, n_time_steps, hidden_layer_size, 4))
    chex.assert_shape(embeds.observed, (batch_size, n_time_steps, hidden_layer_size, 1))


def test_static_covariates_encoder():
    layer = StaticCovariatesEncoder(hidden_layer_size, dropout_rate=0.0, num_static_inputs=2)
    static_context, _ = layer.init_with_output(
        PRNG_KEY, jax.random.uniform(PRNG_KEY, (batch_size, 2, hidden_layer_size)), training=False
    )
    static_context: StaticContextStruct
    # Same shapes as in the original implementation.
    chex.assert_shape(static_context.vector, (batch_size, hidden_layer_size))
    chex.assert_shape(static_context.enrichment, (batch_size, hidden_layer_size))
    chex.assert_shape(static_context.state_c, (batch_size, hidden_layer_size))
    chex.assert_shape(static_context.state_h, (batch_size, hidden_layer_size))


@pytest.mark.parametrize("num_time_steps, features", [(25, 4), (5, 3)])
def test_variable_selection_network(num_time_steps, features):
    layer = VariableSelectionNetwork(
        hidden_layer_size,
        dropout_rate=0,
        num_time_steps=num_time_steps,
        num_inputs=features,
    )

    inputs = jax.random.uniform(PRNG_KEY, (batch_size, num_time_steps, hidden_layer_size, features))
    context = jax.random.uniform(PRNG_KEY, (batch_size, hidden_layer_size))

    (feature, flags, _), _ = layer.init_with_output(PRNG_KEY, inputs, context=context, training=False)
    # Same shapes as in the original implementation.
    chex.assert_shape(feature, (batch_size, num_time_steps, hidden_layer_size))
    chex.assert_shape(flags, (batch_size, num_time_steps, 1, features))


def test_decoder():
    layer = DecoderBlock(num_attention_heads=5, latent_dim=hidden_layer_size, dropout_rate=0)
    decoder_out, _ = layer.init_with_output(
        PRNG_KEY, jax.random.uniform(PRNG_KEY, (batch_size, n_time_steps, hidden_layer_size)), training=False
    )
    chex.assert_shape(decoder_out, (batch_size, n_time_steps, hidden_layer_size))
