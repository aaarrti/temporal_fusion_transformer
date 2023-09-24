from __future__ import annotations

import jax
import jax.numpy as jnp
from absl_extra import logging_utils
from flax.serialization import msgpack_restore
from src.modeling.tft_model import TemporalFusionTransformer
from src.training.training_lib import (
    TrainStateContainer,
    make_optimizer,
    restore_optimizer_state,
    train_step,
)

import temporal_fusion_transformer as tft
from temporal_fusion_transformer.config import get_config
from temporal_fusion_transformer.src.modeling.loss_fn import make_quantile_loss_fn

logging_utils.setup_logging(log_level="INFO")
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_softmax_custom_jvp", True)

# 256 was used on cluster
batch_size = 256


def main():
    config = get_config("electricity")
    with open("fp_error_data.msgpack", "rb") as file:
        byte_data = file.read()

    restored = msgpack_restore(byte_data)

    x_batch = jnp.asarray(restored["x_batch"])
    y_batch = jnp.asarray(restored["y_batch"])

    model = TemporalFusionTransformer.from_config_dict(
        config, data_config=tft.experiments.get_config("electricity"), dtype=jnp.float16
    )
    loss_fn = make_quantile_loss_fn(config.model.quantiles, dtype=jnp.float16)

    state = TrainStateContainer.create(
        tx=make_optimizer(config.optimizer, 18000),
        apply_fn=model.apply,
        params=restored["state"]["params"],
        dropout_key=restored["state"]["dropout_key"],
        loss_fn=loss_fn,
    )
    state = state.replace(step=restored["state"]["step"])
    restored_optimizer = restore_optimizer_state(state.opt_state, restored["state"]["opt_state"])
    state = state.replace(opt_state=restored_optimizer)
    train_step(state, x_batch, y_batch)


if __name__ == "__main__":
    main()
