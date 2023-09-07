import jax.random
import pytest
import flax.linen as nn
import os
import itertools

import numpy as np
import chex
import jax.numpy as jnp
from jax import tree_util
from flax.serialization import msgpack_restore
from typing import Type
from ml_collections import ConfigDict

from temporal_fusion_transformer.src.config_dict import ConfigDictProto
from temporal_fusion_transformer.src.tft_layers import (
    StaticCovariatesEncoder,
    InputEmbedding,
    VariableSelectionNetwork,
    DecoderBlock,
    EmbeddingStruct,
    InputStruct,
)
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer, make_temporal_fusion_transformer
from temporal_fusion_transformer.src.training_lib import single_device_train_step, restore_optimizer_state
from temporal_fusion_transformer.src.datasets.config import get_config

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# This is real config for training, which was failling with NaN.
_CONFIG: ConfigDictProto = ConfigDict(
    {
        "prng_seed": 69,
        "model": {
            "num_attention_heads": 10,
            "num_decoder_blocks": 5,
            "latent_dim": 160,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "optimizer": {
            "clipnorm": 0.0,
            "decay_alpha": 0.1,
            "decay_steps": 0.8,
            "ema": 0.99,
            "learning_rate": 5e-4,
        },
    }
)


@pytest.fixture(scope="module")
def error_causing_state():
    with open("tests/test_data/error_causing_state.msgpack", "rb") as file:
        byte_data = file.read()
    restored = msgpack_restore(byte_data)
    return restored


@pytest.mark.parametrize(
    "dtype",
    [jnp.float16, jnp.bfloat16],
)
def test_mixed_precision(dtype, error_causing_state):
    """We reload real weights and inputs, which caused NaN during fp16 training."""

    x_batch = error_causing_state["x_batch"].astype(dtype)

    model = make_temporal_fusion_transformer(_CONFIG, data_config=get_config("electricity"), dtype=dtype)
    logits = model.apply(
        {"params": error_causing_state["state"]["params"]},
        x_batch,
        True,
        rngs={"dropout": error_causing_state["state"]["dropout_key"]},
    )
    chex.assert_tree_all_finite(logits)
    assert logits.dtype == dtype
    # 3 is default number of quantiles, 24 is TOTAL_TIME_STEPS - num_encoder_steps
    chex.assert_shape(logits, (256, 24, 1, 3))


def test_electricity_model():
    x_batch = jnp.ones((8, 192, 1))
    model = make_temporal_fusion_transformer(_CONFIG, data_config=get_config("electricity"))
    logits, params = model.init_with_output(jax.random.PRNGKey(0), x_batch)
    chex.assert_tree_all_finite(logits)
    # 3 is default number of quantiles, 24 is TOTAL_TIME_STEPS - num_encoder_steps
    chex.assert_shape(logits, (8, 24, 1, 3))
