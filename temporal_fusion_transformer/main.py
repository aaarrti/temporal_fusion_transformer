from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import json
import logging
import shutil
from dataclasses import asdict

import jax
import tensorflow as tf
from absl import app, flags
from keras_core import distribution, mixed_precision
from keras_core.config import disable_traceback_filtering, enable_interactive_logging
from keras_core.utils import set_random_seed
from ml_collections import ConfigDict
from toolz import dicttoolz

import temporal_fusion_transformer as tft
from temporal_fusion_transformer.src.config import Config

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
set_random_seed(33)
jax.config.update("jax_dynamic_shapes", True)
jax.config.update("jax_softmax_custom_jvp", True)
# jax.config.update("jax_log_compiles", True)


tft.setup_logging(log_level="INFO")
FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "task",
    enum_values=["parquet", "dataset", "model", "model_distributed", "hyperparams", "inference"],
    help="Task to run.",
    default=None,
    required=True,
)
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita", "air_passengers"],
    help="Name of the experiment_name.",
    default=None,
    required=True,
)
flags.DEFINE_string(
    "data_dir", default="data", help="Directory into which dataset should be downloaded."
)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations")
flags.DEFINE_boolean("verbose", default=True, help="Verbose mode for training")
flags.DEFINE_integer("prng_seed", default=69, help="PRNG seed")

# For TPU
# mixed_precision.set_global_policy("mixed_bfloat16")
# jax.config.update("jax_default_matmul_precision", "bfloat16")
# For GPU
# mixed_precision.set_global_policy("mixed_float16")
# jax.config.update("jax_default_matmul_precision", "tensorfloat32")


log = logging.getLogger(__name__)


def choose_experiment(name: str) -> tft.experiments.Experiment:
    def default():
        raise RuntimeError("this is unexpected")

    return {
        "electricty": tft.experiments.Electricity,
        # "favorita": tft.experiments.Favorita,
        "air_passengers": tft.experiments.AirPassengers,
    }.get(name, default)()


# --------------------------------------------------------


def parquet_task():
    experiment_name = FLAGS.experiment
    data_dir = FLAGS.data_dir
    ex = choose_experiment(experiment_name)
    ex.dataset_cls().convert_to_parquet(f"{data_dir}/{experiment_name}")


def dataset_task():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"
    ex = choose_experiment(experiment_name)
    ex.dataset_cls().make_dataset(data_dir, save_dir=data_dir)


# --------------------------------------------------------


def model_task():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    # shutil.rmtree(
    #    "/Users/artemsereda/Documents/IdeaProjects/temporal_fusion_transformer/data/xla_logs/",
    # )

    ex = choose_experiment(experiment_name)
    ex.train_model(
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        verbose="auto" if FLAGS.verbose else 1,
        data_dir=data_dir,
        jit_compile=False,
        save_filename=f"models/{experiment_name}/weights.h5",
    )


def model_distributed_task():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    mesh = distribution.DataParallel()
    distribution.set_distribution(mesh)

    ex = choose_experiment(experiment_name)
    ex.train_model(
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        verbose="auto" if FLAGS.verbose else 1,
        data_dir=data_dir,
    )


def main(_):
    log.info("-" * 50)
    log.info(f"TF devices = {tf.config.get_visible_devices()}")
    log.info(f"JAX devices = {jax.devices()}")

    def map_fn(v):
        if isinstance(v, (ConfigDict, Config)):
            if isinstance(v, ConfigDict):
                v = v.to_dict()
            if isinstance(v, Config):
                v = asdict(v)
            return dicttoolz.valmap(map_fn, v)
        else:
            return v

    absl_flags = dicttoolz.valmap(map_fn, flags.FLAGS.flag_values_dict())

    log.info(f"ABSL flags: {json.dumps(absl_flags, sort_keys=True, indent=4)}")
    log.info("-" * 50)

    return {
        "model": model_task,
        "parquet": parquet_task,
        "dataset": dataset_task,
        "model_distributed": model_distributed_task,
    }[FLAGS.task]()


if __name__ == "__main__":
    app.run(main)
