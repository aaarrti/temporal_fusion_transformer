from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
from absl import flags
from absl_extra import tasks, logging_utils, notifier
from ml_collections import config_flags
import platform
import os

import temporal_fusion_transformer as tft

# For debugging
import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_disable_jit", True)

# fmt: off
FLAGS = flags.FLAGS
flags.DEFINE_enum("experiment", enum_values=["electricity", "favorita"], help="Name of the experiment_name", default=None, required=True)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_string("data_dir", help="Data directory", default="data")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations.")
flags.DEFINE_boolean("jit_module", default=False, help="Apply nn.jit to model")
flags.DEFINE_boolean("full_reshuffle", default=False, help="Fully reshuffle dataset before training")
CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
# fmt: on
logging_utils.setup_logging(log_level="INFO")


def make_notifier():
    if platform.system().lower() == "linux":
        return notifier.SlackNotifier(
            slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["SLACK_CHANNEL_ID"]
        )
    else:
        return notifier.NoOpNotifier()


@tasks.register_task(notifier=make_notifier)
def train_model():
    data_dir, experiment, epochs, batch_size, mixed_precision, jit_module, full_reshuffle = (
        FLAGS.data_dir,
        FLAGS.experiment,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.mixed_precision,
        FLAGS.jit_module,
        FLAGS.full_reshuffle,
    )
    config = CONFIG.value

    tft.training_scripts.train_experiment(
        data_dir=data_dir,
        experiment_name=experiment,
        epochs=epochs,
        batch_size=batch_size,
        config=config,
        mixed_precision=mixed_precision,
        jit_module=jit_module,
        prefetch_buffer_size=4,
        device_type="gpu",
        save_path="model.msgpack",
        full_reshuffle=full_reshuffle,
    )


if __name__ == "__main__":
    tasks.run()