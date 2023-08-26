from __future__ import annotations

import jax
import jax.numpy as jnp
from absl import logging
from absl_extra import logging_utils
from flax.serialization import msgpack_restore
from jax import tree_util
from sklearn.utils import gen_batches
from tqdm.auto import tqdm
from flax.training.dynamic_scale import DynamicScale

from temporal_fusion_transformer.config import get_config
from temporal_fusion_transformer.src.quantile_loss import make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_model import InputStruct, TemporalFusionTransformer
from temporal_fusion_transformer.src.training_lib import (
    TrainStateContainer,
    make_optimizer,
    restore_optimizer_state,
    single_device_train_step,
)

# CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
logging_utils.setup_logging(log_level="INFO")
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_softmax_custom_jvp", True)

# 256 was used on cluster, smaller batch size cause no problems at all
batch_size = 8


def unshard(batch: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(batch, [-1, *batch.shape[2:]])


def main():
    config = get_config("electricity")
    with open("fp_error.msgpack", "rb") as file:
        byte_data = file.read()

    restored = msgpack_restore(byte_data)

    x_batch = restored["x_batch"]
    y_batch = restored["y_batch"]

    # y_batch = unshard(y_batch)
    # x_batch = tree_util.tree_map(unshard, x_batch)
    # num_shards = len(y_batch)

    model = TemporalFusionTransformer.from_config_dict(config, dtype=jnp.float16)
    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles, dtype=jnp.float16)

    state = TrainStateContainer.create(
        tx=make_optimizer(config.optimizer, 18000, 1),
        apply_fn=model.apply,
        params=restored["state"]["params"],
        dropout_key=restored["state"]["dropout_key"],
        loss_fn=loss_fn,
    )
    state = state.replace(step=restored["state"]["step"])
    restored_optimizer = restore_optimizer_state(state.opt_state, restored["state"]["opt_state"])
    state = state.replace(opt_state=restored_optimizer)

    batches = gen_batches(len(y_batch), batch_size)
    num_mini_batches = len(list(gen_batches(len(y_batch), batch_size)))
    # step_fn = logging_utils.log_exception(single_device_train_step, ignore_argnums=(0,))
    pbar = tqdm(total=num_mini_batches)

    for i, batch in enumerate(batches):
        pbar.update(1)

        x = tree_util.tree_map(lambda k: k[batch.start : batch.stop], x_batch)
        x = InputStruct(
            static=x["static"],
            known_categorical=x["known_categorical"],
            known_real=x["known_real"],
            observed=None,
            unknown=None,
        ).cast_inexact(jnp.float16)
        y = y_batch[batch.start : batch.stop].astype(jnp.float16)
        # try:
        state, _ = single_device_train_step(state, x, y)
        # except FloatingPointError as err:
        #     logging.error(err)


if __name__ == "__main__":
    main()
