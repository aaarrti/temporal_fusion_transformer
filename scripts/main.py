from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
from absl import flags
from absl import logging
from absl_extra import tasks, logging_utils
from ml_collections import config_flags

import temporal_fusion_transformer as tft

# For debugging
import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_disable_jit", True)

# fmt: off
FLAGS = flags.FLAGS
flags.DEFINE_enum("experiment", enum_values=["electricity", "favorita"], help="Name of the experiment_name", default=None, required=True)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_string("data_dir", help="Data directory", default="data")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations")
flags.DEFINE_boolean("jit_module", default=False, help="Apply nn.jit to model")
flags.DEFINE_boolean("full_reshuffle", default=False, help="Fully reshuffle dataset before training")
flags.DEFINE_boolean("profile", default=False, help="Run with profiling")
flags.DEFINE_boolean("verbose", default=True, help="Verbose mode for training")
flags.DEFINE_string("save_path", default="model.msgpack", help="Save path for model")
flags.DEFINE_integer("prefetch_buffer_size", default=2, help="Prefetch buffer size")
CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
# fmt: on
logging_utils.setup_logging(log_level="DEBUG")


@tasks.register_task(name="data")
def make_dataset():
    experiment_factories = {
        "electricity": tft.datasets.Electricity,
        "favorita": tft.datasets.Favorita,
    }
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"
    experiment: tft.datasets.MultiHorizonTimeSeriesDataset = experiment_factories[experiment_name]()
    (train_ds, val_ds, test_ds), feature_space = experiment.make_dataset(data_dir)
    logging.info(f"Saving training split")
    train_ds.save(f"{data_dir}/training", compression="GZIP")
    logging.info(f"Saving validation split")
    val_ds.save(f"{data_dir}/validation", compression="GZIP")
    logging.info(f"Saving test split")
    test_ds.save(f"{data_dir}/test", compression="GZIP")
    feature_space.save(f"{data_dir}/features_space.keras")


# def make_notifier():
#    if platform.system().lower() == "linux":
#        return notifier.SlackNotifier(
#            slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["SLACK_CHANNEL_ID"]
#        )
#    else:
#        return notifier.NoOpNotifier()


@tasks.register_task(name="hyperparams")
def train_model():
    tft.hyperparams.optimize_experiment_hyperparams(
        data_dir=FLAGS.data_dir,
        experiment_name=FLAGS.experiment,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        config=CONFIG.value,
        mixed_precision=FLAGS.mixed_precision,
        jit_module=FLAGS.jit_module,
        device_type="gpu",
        verbose=FLAGS.verbose,
    )


@tasks.register_task(name="model")
def train_model():
    tft.training_scripts.train_experiment(
        data_dir=FLAGS.data_dir,
        experiment_name=FLAGS.experiment,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        config=CONFIG.value,
        mixed_precision=FLAGS.mixed_precision,
        jit_module=FLAGS.jit_module,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        device_type="gpu",
        save_path=FLAGS.save_path,
        full_reshuffle=FLAGS.full_reshuffle,
        profile=FLAGS.profile,
        verbose=FLAGS.verbose,
    )


if __name__ == "__main__":
    tasks.run()
