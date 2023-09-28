from __future__ import annotations

import os

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import platform

# For debugging
import jax
from absl import flags, logging
from absl_extra.cuda_utils import supports_mixed_precision
from absl_extra.logging_utils import setup_logging
from absl_extra.notifier import NoOpNotifier, SlackNotifier
from absl_extra.tasks import register_task, run
from ml_collections import config_flags

import temporal_fusion_transformer as tft

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_softmax_custom_jvp", True)
# jax.config.update("jax_default_matmul_precision", "tensorfloat32")
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_disable_jit", True)

# fmt: off
FLAGS = flags.FLAGS
flags.DEFINE_enum("experiment", enum_values=["electricity", "favorita"], help="Name of the experiment_name",
                  default=None, required=True)
flags.DEFINE_integer("batch_size", default=8, help="Training batch size")
flags.DEFINE_string("data_dir", help="Data directory", default="data")
flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train.")
flags.DEFINE_boolean("mixed_precision", default=False, help="Use mixed (b)float16 for computations")
flags.DEFINE_boolean("jit_module", default=False, help="Apply nn.jit to model")
flags.DEFINE_boolean("profile", default=False, help="Run with profiling")
flags.DEFINE_boolean("verbose", default=True, help="Verbose mode for training")
# fmt: on

setup_logging(log_level="INFO")


def make_notifier() -> SlackNotifier | None:
    if platform.system().lower() == "linux":
        return SlackNotifier(slack_token=os.environ["SLACK_BOT_TOKEN"], channel_id=os.environ["SLACK_CHANNEL_ID"])
    else:
        return NoOpNotifier()


@register_task(name="model", notifier=make_notifier)
def train_model():
    experiment_name = FLAGS.experiment
    mixed_precision = FLAGS.mixed_precision and supports_mixed_precision()

    if experiment_name == "electricity":
        trainer = tft.experiments.Electricity().trainer
    elif experiment_name == "favorita":
        trainer = None
    else:
        raise RuntimeError("this is unexpected")

    trainer.run(
        data_dir=f"{FLAGS.data_dir}/{experiment_name}",
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        mixed_precision=mixed_precision,
        jit_module=FLAGS.jit_module,
        verbose=FLAGS.verbose,
    )


@register_task(name="model_distributed", notifier=make_notifier)
def train_model_dsitributed():
    experiment_name = FLAGS.experiment
    mixed_precision = FLAGS.mixed_precision and supports_mixed_precision()

    if experiment_name == "electricity":
        trainer = tft.experiments.Electricity().trainer
    elif experiment_name == "favorita":
        trainer = None
    else:
        raise RuntimeError("this is unexpected")

    trainer.run_distributed(
        data_dir=f"{FLAGS.data_dir}/{experiment_name}",
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        mixed_precision=mixed_precision,
        jit_module=FLAGS.jit_module,
        verbose=FLAGS.verbose,
        device_type="gpu",
    )


if __name__ == "__main__":
    run()
