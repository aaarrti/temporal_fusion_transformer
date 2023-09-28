from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import chex
import flax.jax_utils
import jax.numpy as jnp
import jax.random
import ml_collections
import optax
import pytest
from flax.serialization import msgpack_restore

from temporal_fusion_transformer.src.experiments.config import get_config
from temporal_fusion_transformer.src.modeling.loss_fn import make_quantile_loss_fn
from temporal_fusion_transformer.src.modeling.tft_model import (
    make_temporal_fusion_transformer,
)
from temporal_fusion_transformer.src.training.training import make_attention_mesh
from temporal_fusion_transformer.src.training.training_lib import (
    TrainStateContainer,
    distributed_validation_step,
)

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.config_dict import ConfigDict

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# This is real config for training, which was failling with NaN.
_CONFIG: ConfigDict = ml_collections.ConfigDict(
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
    x_batch = jnp.ones((8, 192, 5))
    model = make_temporal_fusion_transformer(_CONFIG, data_config=get_config("electricity"))
    logits, params = model.init_with_output(jax.random.PRNGKey(0), x_batch)
    chex.assert_tree_all_finite(logits)
    # 3 is default number of quantiles, 24 is TOTAL_TIME_STEPS - num_encoder_steps
    chex.assert_shape(logits, (8, 24, 1, 3))


# ------------- this can be run only once ----

n = 4


@pytest.fixture(scope="function")
def force_multi_device():
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n}"
    yield
    del os.environ["XLA_FLAGS"]


def test_partition_attention_kernel(force_multi_device, capsys):
    assert jax.device_count() == n

    x_sharded = jnp.ones((n, 8, 192, 5))
    y_sharded = jnp.ones((n, 8, 24, 1))

    model = make_temporal_fusion_transformer(
        _CONFIG, data_config=get_config("electricity"), attention_mesh=make_attention_mesh()
    )

    params = model.init(jax.random.PRNGKey(0), x_sharded[0])["params"]

    with capsys.disabled():
        jax.debug.visualize_array_sharding(params["decoder_blocks_0"]["SelfAttention_0"]["key"]["kernel"].value)

    state = TrainStateContainer.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.sgd(1e-3),
        loss_fn=make_quantile_loss_fn(quantiles=_CONFIG.model.quantiles),
        dropout_key=jax.random.PRNGKey(0),
    )

    state = flax.jax_utils.replicate(state)

    distributed_validation_step(state, x_sharded, y_sharded)
