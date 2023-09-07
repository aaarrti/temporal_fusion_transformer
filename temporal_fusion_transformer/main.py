from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
# For debugging
import jax
from absl import flags, logging
from absl_extra.logging_utils import setup_logging
from absl_extra.tasks import register_task, run
from ml_collections import config_flags

import temporal_fusion_transformer as tft

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
flags.DEFINE_boolean("profile", default=False, help="Run with profiling")
flags.DEFINE_boolean("verbose", default=True, help="Verbose mode for training")
flags.DEFINE_string("save_path", default="model.msgpack", help="Save data_dir for model")
flags.DEFINE_integer("prefetch_buffer_size", default=0, help="Prefetch buffer size")
CONFIG = config_flags.DEFINE_config_file("config", default="temporal_fusion_transformer/config.py")
# fmt: on
setup_logging(log_level="INFO")


@register_task(name="data")
def make_dataset():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    if experiment_name == "electricity":
        # fmt: off
        (train_dataset, validation_dataset, test_dataset), preprocessor = tft.datasets.electricity.make_dataset(
            data_dir)
        # fmt: on
    elif experiment_name == "favorita":
        # fmt: off
        (train_dataset, validation_dataset, test_dataset), preprocessor = tft.datasets.favorita.make_dataset(data_dir)
        # fmt: on
    else:
        raise RuntimeError("this is unexpected")

    logging.info(f"Saving training split")
    train_dataset.save(f"{data_dir}/training", compression="GZIP")
    logging.info(f"Saving validation split")
    validation_dataset.save(f"{data_dir}/validation", compression="GZIP")
    logging.info(f"Saving test split")

    test_dataset.save(f"{data_dir}/test", compression="GZIP")
    tft.datasets.serialize_preprocessor(preprocessor, data_dir)


@register_task(name="model")
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


@register_task(name="model")
def train_model():
    experiment_name = FLAGS.experiment

    data_config = tft.datasets.get_config(experiment_name)
    tft.training_scripts.train_experiment(
        data_dir=FLAGS.data_dir,
        experiment_name=experiment_name,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        config=CONFIG.value,
        data_config=data_config,
        mixed_precision=FLAGS.mixed_precision,
        jit_module=FLAGS.jit_module,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        device_type="gpu",
        save_path=FLAGS.save_path,
        profile=FLAGS.profile,
        verbose=FLAGS.verbose,
    )


if __name__ == "__main__":
    run()
