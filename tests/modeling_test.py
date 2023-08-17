import jax.random
import pytest
import flax.linen as nn
import os

import numpy as np
import chex
import jax.numpy as jnp
from typing import Type

from temporal_fusion_transformer.src.tft_layers import (
    StaticCovariatesEncoder,
    InputEmbedding,
    VariableSelectionNetwork,
    DecoderBlock,
    EmbeddingStruct,
    InputStruct,
)
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5

PRNG_KEY = jax.random.PRNGKey(0)


def make_model(module: Type[nn.Module], **kwargs) -> nn.Module:
    return module(
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
        **kwargs,
    )


@pytest.mark.parametrize("jit", [False, True])
def test_jit_model(jit):
    x_batch = jax.random.uniform(PRNG_KEY, (8, 30, 5))

    module_type = TemporalFusionTransformer
    if jit:
        module_type = nn.jit(module_type)

    model: nn.Module = make_model(module_type)
    logits, _ = model.init_with_output(PRNG_KEY, x_batch)
    chex.assert_tree_all_finite(logits)
    # 3 is default number of quantiles.
    chex.assert_shape(logits, (batch_size, hidden_layer_size, 3))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_mixed_precision(dtype):
    x_batch = jax.random.uniform(PRNG_KEY, (8, 30, 5)).astype(dtype)
    model: nn.Module = make_model(TemporalFusionTransformer, dtype=dtype)
    logits, _ = model.init_with_output(PRNG_KEY, x_batch)
    chex.assert_tree_all_finite(logits)
    assert logits.dtype == dtype
    # 3 is default number of quantiles.
    chex.assert_shape(logits, (batch_size, hidden_layer_size, 3))
