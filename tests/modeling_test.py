import jax.random
import pytest
import flax.linen as nn
import os

import numpy as np

from temporal_fusion_transformer.src.tft_layers import (
    StaticCovariatesEncoder,
    InputEmbedding,
    VariableSelectionNetwork,
    DecoderBlock,
    EmbeddingStruct,
    InputStruct,
)
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.global_config import GlobalConfig

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
    # Same shapes as in the original implementation.
    assert (batch_size, 2, hidden_layer_size) == embeds.static.shape
    assert (batch_size, n_time_steps, hidden_layer_size, 4) == embeds.known.shape
    assert (batch_size, n_time_steps, hidden_layer_size, 1) == embeds.observed.shape


def test_static_covariates_encoder():
    layer = StaticCovariatesEncoder(hidden_layer_size, dropout_rate=0.0, num_static_inputs=2)
    static_context, _ = layer.init_with_output(
        PRNG_KEY, jax.random.uniform(PRNG_KEY, (batch_size, 2, hidden_layer_size)), training=False
    )
    # Same shapes as in the original implementation.
    assert (batch_size, hidden_layer_size) == static_context.vector.shape
    assert (batch_size, hidden_layer_size) == static_context.enrichment.shape
    assert (batch_size, hidden_layer_size) == static_context.state_h.shape
    assert (batch_size, hidden_layer_size) == static_context.state_c.shape


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
    assert (batch_size, num_time_steps, hidden_layer_size) == feature.shape
    assert (batch_size, num_time_steps, 1, features == flags.shape)


def test_decoder():
    layer = DecoderBlock(num_attention_heads=5, latent_dim=hidden_layer_size, dropout_rate=0)
    decoder_out, _ = layer.init_with_output(
        PRNG_KEY, jax.random.uniform(PRNG_KEY, (batch_size, n_time_steps, hidden_layer_size)), training=False
    )
    assert (batch_size, n_time_steps, hidden_layer_size) == decoder_out.shape


@pytest.mark.parametrize("jit", [False, True])
def test_tft_model(jit):
    GlobalConfig.update(jit_module=jit)
    x_batch = np.random.default_rng(69).uniform(size=(8, 30, 5))
    model: nn.Module = TemporalFusionTransformer(
        num_encoder_steps=25,
        num_attention_heads=5,
        latent_dim=hidden_layer_size,
        static_categories_sizes=static_categories_sizes,
        known_categories_sizes=known_categories_sizes,
        dropout_rate=0,
        num_quantiles=3,
        num_outputs=1,
        num_decoder_blocks=1,
        input_known_real_idx=[0, 1],
        input_static_idx=[2],
        input_known_categorical_idx=[3],
        input_observed_idx=[4],
        total_time_steps=30,
    )
    logits, _ = model.init_with_output(PRNG_KEY, x_batch)
    logits = logits
    assert not np.isnan(logits).any()
    # 3 is default number of quantiles.
    assert (batch_size, 5, 3) == logits.shape


import tensorflow as tf

tf.keras.layers.MultiHeadAttention
