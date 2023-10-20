from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
from absl import app, flags
from keras_core import distribution, mixed_precision
import logging
import json
import temporal_fusion_transformer as tft

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
jax.config.update("jax_softmax_custom_jvp", True)
jax.config.update("jax_default_dtype_bits", "32")


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
    enum_values=["electricity", "favorita"],
    help="Name of the experiment_name.",
    default=None,
    required=True,
)
flags.DEFINE_string("data_dir", default="data", help="Directory into which dataset should be downloaded.")
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
    if name == "electricity":
        return tft.experiments.Electricity()
    elif name == "favorita":
        return tft.experiments.Favorita()
    else:
        raise RuntimeError("this is unexpected")


# --------------------------------------------------------


def parquet():
    experiment_name = FLAGS.experiment
    data_dir = FLAGS.data_dir
    ex = choose_experiment(experiment_name)
    ex.dataset().convert_to_parquet(f"{data_dir}/{experiment_name}")


def dataset():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"
    ex = choose_experiment(experiment_name)
    ex.dataset().make_dataset(data_dir, save_dir=data_dir)


# --------------------------------------------------------


def model():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    ex = choose_experiment(experiment_name)
    ex.train_model(
        epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose="auto" if FLAGS.verbose else 1, data_dir=data_dir
    )


def model_distributed():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"

    mesh = distribution.DataParallel()
    distribution.set_distribution(mesh)

    ex = choose_experiment(experiment_name)
    ex.train_model(
        epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, verbose="auto" if FLAGS.verbose else 1, data_dir=data_dir
    )


def select_task(_):
    log.info("-" * 50)
    log.info(f"JAX devices = {jax.devices()}")
    log.info(f"ABSL flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}")
    log.info("-" * 50)
    
    return {"model": model, "parquet": parquet, "dataset": dataset, "model_distributed": model_distributed}[
        FLAGS.task
    ]()


if __name__ == "__main__":
    app.run(select_task)
