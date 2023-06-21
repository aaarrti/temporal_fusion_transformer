from __future__ import annotations

import functools
import logging
from typing import Dict, Tuple, Callable
import datetime
import platform

import tensorflow as tf
from absl import flags

from absl_extra import (
    register_task,
    run,
    requires_gpu,
    supports_mixed_precision,
    setup_logging,
)
from keras.api.keras.experimental import CosineDecay
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.utils.tf_utils import set_random_seed
from temporal_fusion_transformer import make_tft_model, experiments, can_jit_compile

minor_tf_api_version = int(tf.__version__.split(".")[1])

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita"],
    help="Name of the experiment",
    default="electricity",
)
flags.DEFINE_integer("batch_size", default=64, help="Training batch size")
flags.DEFINE_integer("epochs", default=1, help="Number of training epochs")
flags.DEFINE_string("data_dir", help="Data directory", default="datasets")
flags.DEFINE_string("logs_dir", help=None, default="logs")

PRNG_SEED = 42
set_random_seed(PRNG_SEED)
setup_logging()
if supports_mixed_precision():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
if can_jit_compile():
    tf.config.optimizer.set_jit("autoclustering")


def electricity_map_fn(
    arg: Dict[str, tf.Tensor], dtype
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    return (
        dict(
            static=tf.cast(arg["inputs_static"], tf.int32),
            known_real=tf.cast(arg["inputs_known_real"], dtype),
        ),
        tf.cast(arg["outputs"], dtype),
    )


def make_optimizer(learning_rate) -> tf.keras.optimizers.Optimizer:
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        from keras.optimizers.legacy.adam import Adam

        return Adam(learning_rate)

    if minor_tf_api_version >= 11:
        from keras.optimizers.adam import Adam
    else:
        from keras.optimizers.optimizer_experimental.adam import Adam  # noqa

    return Adam(learning_rate, jit_compile=True)


def prepare_dataset(
    ds: tf.data.Dataset,
    batch_size: int,
    epochs: int,
    map_fn: Callable[[Dict[str, tf.Tensor]], Tuple[Dict[str, tf.Tensor], tf.Tensor]],
) -> tf.data.Dataset:
    ds = (
        ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(32, reshuffle_each_iteration=True, seed=PRNG_SEED)
        .cache()
        .repeat(epochs)
        .apply(tf.data.experimental.prefetch_to_device("/gpu:0", tf.data.AUTOTUNE))
    )
    return ds


@register_task
@requires_gpu
def main():
    """
    1. The parameters were picked out of the blue, this script's purpose is to demonstrate APIs.
    2. The script was specifically designed to run in NVIDIA's TensorFlow container with tag 22.12-tf2-py3,
    more about it here https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow.
    """
    logging.info(f"{tf.config.list_physical_devices() = }")
    experiment_name = FLAGS.experiment
    logs_dir = FLAGS.logs_dir

    if experiment_name == "electricity":
        experiment = experiments.electricity_experiment
        map_fn = electricity_map_fn

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir

    training_split = tf.data.Dataset.load(f"{data_dir}/{experiment_name}/training/")
    validation_split = tf.data.Dataset.load(f"{data_dir}/{experiment_name}/validation/")
    num_train_samples = int(training_split.cardinality())
    num_validation_samples = int(validation_split.cardinality())

    steps_per_epoch = num_train_samples // batch_size
    val_steps = num_validation_samples // batch_size

    map_fn = functools.partial(
        map_fn, dtype=tf.keras.mixed_precision.global_policy().compute_dtype  # noqa
    )

    training_split = prepare_dataset(training_split, batch_size, epochs, map_fn)
    validation_split = prepare_dataset(validation_split, batch_size, epochs, map_fn)

    model = make_tft_model(
        experiment,  # noqa
        # Those were picked randomly tbh.
        num_attention_heads=4,
        hidden_layer_size=60,
        num_stacks=4,
    )
    model.compile(
        optimizer=make_optimizer(
            # Also picked randomly lol.
            CosineDecay(
                5e-3,
                int(steps_per_epoch * epochs),
                alpha=0.02,
            ),
        ),
        # We can't use XLA and CuDNN at the same time.
        jit_compile=False,
    )
    model.fit(
        training_split,
        epochs=epochs,
        validation_data=validation_split,
        callbacks=[
            TensorBoard(
                f"{logs_dir}/{experiment_name}/tensorboard_logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
                update_freq=10,
                # Graph is pretty useless, unless debugging NaN's.
                write_graph=False,
                # Profile however, does provide some really helpfully details.
                # profile_batch=True,
                write_steps_per_second=True,
            ),
            TerminateOnNaN(),
            # No need really, unless running super large scale training.
            # BackupAndRestore(f"{logs_dir}/{experiment_name}/checkpoints"),
        ],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )

    with tf.device("/cpu:0"):
        model.save_weights(f"{logs_dir}/{experiment_name}/weights_v1")


if __name__ == "__main__":
    run("train_TFT")
