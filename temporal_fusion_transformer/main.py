from __future__ import annotations

import tensorflow as tf
import jax
from absl import flags, logging
from absl_extra import tasks, logging_utils
from ml_collections import config_flags

import temporal_fusion_transformer as tft
from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset
from temporal_fusion_transformer.src.datasets.electricity import Electricity
from temporal_fusion_transformer.src.datasets.favorita import Favorita

tf.config.set_visible_devices([], "GPU")

# tft.GlobalConfig().update(jit_module=True)
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_nans", True)
# For debugging
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_disable_jit", True)

# fmt: off
FLAGS = flags.FLAGS
flags.DEFINE_enum("experiment", enum_values=["electricity", "favorita"], help="Name of the experiment", default=None, required=True)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_string("data_dir", help="Data directory", default="data")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations.")
flags.DEFINE_boolean("jit_module", default=True, help="Apply nn.jit to model")
CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
# fmt: on
logging_utils.setup_logging(log_level="INFO")


_experiment_factories = {
    "electricity": Electricity,
    "favorita": Favorita,
}


@tasks.register_task(name="data")
def make_dataset_task():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment

    data_dir = f"{data_dir}/{experiment_name}"
    experiment: MultiHorizonTimeSeriesDataset = _experiment_factories[experiment_name]()
    (train_ds, val_ds, test_ds), feature_space = experiment.make_dataset(data_dir)
    logging.info(f"Saving training split")
    train_ds.save(f"{data_dir}/training", compression="GZIP")
    logging.info(f"Saving validation split")
    val_ds.save(f"{data_dir}/validation", compression="GZIP")
    logging.info(f"Saving test split")
    test_ds.save(f"{data_dir}/test", compression="GZIP")
    feature_space.save(f"{data_dir}/features_space.keras")


@tasks.register_task(name="model")
def train_model():
    data_dir, experiment, epochs, batch_size, mixed_precision, jit_module = (
        FLAGS.data_dir,
        FLAGS.experiment,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.mixed_precision,
        FLAGS.jit_module,
    )
    config = CONFIG.value
    tft.training.train_on_single_device(
        data_dir=data_dir,
        experiment_name=experiment,
        epochs=epochs,
        batch_size=batch_size,
        config=config,
        mixed_precision=mixed_precision,
        jit_module=jit_module,
    )


@tasks.register_task(name="model_distributed")
def train_model():
    data_dir, experiment, epochs, batch_size, mixed_precision, jit_module = (
        FLAGS.data_dir,
        FLAGS.experiment,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.mixed_precision,
        FLAGS.jit_module,
    )
    config = CONFIG.value
    tft.training.train_on_multiple_devices(
        data_dir=data_dir,
        experiment_name=experiment,
        epochs=epochs,
        batch_size=batch_size,
        config=config,
        mixed_precision=mixed_precision,
        jit_module=jit_module,
    )


if __name__ == "__main__":
    tasks.run()
