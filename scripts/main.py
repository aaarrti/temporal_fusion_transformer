from __future__ import annotations

import jax
from absl import flags
from absl_extra import tasks, logging_utils
from ml_collections import config_flags

import temporal_fusion_transformer as tft

# tft.GlobalConfig().update(jit_module=True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_nans", True)
# For debugging
# jax.config.update("jax_log_compiles", True)
jax.config.update("jax_disable_jit", True)

# fmt: off
FLAGS = flags.FLAGS
flags.DEFINE_enum("experiment", enum_values=["electricity", "favorita"], help="Name of the experiment", default=None, required=True)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_string("data_dir", help="Data directory", default="data")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations.")
CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
# fmt: on
logging_utils.setup_logging(log_level="INFO")


@tasks.register_task(name="data")
def make_dataset_task():
    data_dir, experiment = FLAGS.data_dir, FLAGS.experiment
    tft.scripts.make_dataset(data_dir, experiment)


@tasks.register_task(name="model")
def train_model():
    data_dir, experiment, epochs, batch_size, mixed_precision = (
        FLAGS.data_dir,
        FLAGS.experiment,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.mixed_precision,
    )
    config = CONFIG.value
    tft.scripts.train_on_single_device(
        data_dir=data_dir,
        experiment_name=experiment,
        epochs=epochs,
        batch_size=batch_size,
        config=config,
        mixed_precision=mixed_precision,
    )


if __name__ == "__main__":
    tasks.run()
