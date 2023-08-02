import pytest

import os

import numpy as np

from temporal_fusion_transformer.src.tft_layers import (
    StaticCovariatesEncoder,
    InputEmbedding,
    VariableSelectionNetwork,
    DecoderBlock,
    InputStruct,
)

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5


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
    known, observed, static = layer(x_batch)
    # Same shapes as in the original implementation.
    assert (batch_size, 2, hidden_layer_size) == static.shape
    assert (batch_size, n_time_steps, hidden_layer_size, 4) == known.shape
    assert (batch_size, n_time_steps, hidden_layer_size, 1) == observed.shape


def test_static_covariates_encoder():
    layer = StaticCovariatesEncoder(hidden_layer_size, dropout_rate=0.0, prng_seed=69, num_static_inputs=2)
    static_context = layer(np.random.default_rng(69).uniform(size=(batch_size, 2, hidden_layer_size)))
    # Same shapes as in the original implementation.
    assert (batch_size, hidden_layer_size) == static_context["vector"].shape
    assert (batch_size, hidden_layer_size) == static_context["enrichment"].shape
    assert (batch_size, hidden_layer_size) == static_context["state_h"].shape
    assert (batch_size, hidden_layer_size) == static_context["state_c"].shape


@pytest.mark.parametrize("num_time_steps, features", [(25, 4), (5, 3)])
def test_variable_selection_network(num_time_steps, features):
    layer = VariableSelectionNetwork(
        hidden_layer_size, prng_seed=69, dropout_rate=0, num_time_steps=num_time_steps, num_inputs=features
    )

    inputs = np.random.default_rng(69).uniform(size=(batch_size, num_time_steps, hidden_layer_size, features))
    context = np.random.default_rng(69).uniform(size=(batch_size, hidden_layer_size))

    feature, flags, _ = layer(inputs, context=context)
    # Same shapes as in the original implementation.
    assert (batch_size, num_time_steps, hidden_layer_size) == feature.shape
    assert (batch_size, num_time_steps, 1, features == flags.shape)


def test_decoder():
    layer = DecoderBlock(4, latent_dim=hidden_layer_size, dropout_rate=0, prng_seed=69)
    decoder_out = layer(
        np.random.default_rng(69).uniform(size=(batch_size, n_time_steps, hidden_layer_size)),
    )
    assert (batch_size, n_time_steps, hidden_layer_size) == decoder_out.shape
