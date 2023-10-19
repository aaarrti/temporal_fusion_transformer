from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
from absl import app, flags, logging
from absl_extra.logging_utils import setup_logging
from absl_extra.notifier import NoOpNotifier
from absl_extra.tasks import register_task
from absl_extra.tasks import run as main
from keras_core import mixed_precision

import temporal_fusion_transformer as tft

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
jax.config.update("jax_softmax_custom_jvp", True)
jax.config.update("jax_default_dtype_bits", "32")
# for cuda
# jax.config.update("jax_default_matmul_precision", "tensorfloat32")
# jax.config.update("jax_default_matmul_precision", "bfloat16")

# setup_logging()
setup_logging(log_level="INFO")
FLAGS = flags.FLAGS
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


# mixed_precision.set_global_policy("mixed_float16")


def choose_experiment(name: str) -> tft.experiments.Experiment:
    if name == "electricity":
        return tft.experiments.Electricity()
    elif name == "favorita":
        return tft.experiments.Favorita()
    else:
        raise RuntimeError("this is unexpected")


# --------------------------------------------------------


@register_task(name="parquet", notifier=NoOpNotifier())
def parquet():
    experiment_name = FLAGS.experiment
    data_dir = FLAGS.data_dir
    ex = choose_experiment(experiment_name)
    ex.dataset().convert_to_parquet(f"{data_dir}/{experiment_name}")


@register_task(name="dataset", notifier=NoOpNotifier())
def dataset():
    data_dir, experiment_name = FLAGS.data_dir, FLAGS.experiment
    data_dir = f"{data_dir}/{experiment_name}"
    ex = choose_experiment(experiment_name)
    ex.dataset().make_dataset(data_dir, save_dir=data_dir)


# --------------------------------------------------------


@register_task(name="model", notifier=NoOpNotifier())
def model():
    experiment_name = FLAGS.experiment
    ex = choose_experiment(experiment_name)
    ex.train_model(
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        verbose="auto",
    )


# @register_task(name="model_distributed", notifier=NoOpNotifier())
# def model_distributed(_):
#    pass


# --------------------------------------------------------

# @register_task(name="hyperparams", notifier=NoOpNotifier())
# def hyperparams(_):
#    pass


if __name__ == "__main__":
    main()
