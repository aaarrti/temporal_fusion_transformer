from __future__ import annotations

import functools
import logging
from typing import Dict, Tuple, Callable
import datetime
import platform

import tensorflow as tf
from keras.api.keras.experimental import CosineDecay
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.utils.tf_utils import set_random_seed
import temporal_fusion_transformer as tft
from absl import flags
from absl_extra import (
    register_task,
    run,
    requires_gpu,
    supports_mixed_precision,
    setup_logging,
)

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
flags.DEFINE_string("logs_dir", help="Tensorboard logs directory", default="logs")
flags.DEFINE_integer("num_attention_heads", default=4, help="Number of attention heads")
flags.DEFINE_integer("hidden_size", default=60, help="Hidden layer size")
flags.DEFINE_integer("num_stacks", default=2, help="Number of encoder stacks")
flags.DEFINE_integer("prng_seed", default=42, help="Random number generator seed")
flags.DEFINE_float("initial_learning_rate", default=5e-3, help="Initial learning rate")
flags.DEFINE_float(
    "decay_alpha",
    default=0.02,
    help="Minimal portion of inital learning rate until which decays.",
)
flags.DEFINE_integer("log_frequency", default=10, help="TensorBoard log frequency")


setup_logging()
if supports_mixed_precision():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
if tft.can_jit_compile():
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
        .shuffle(32, reshuffle_each_iteration=True, seed=FLAGS.prng_seed)
        .cache()
        .repeat(epochs)
        .apply(tf.data.experimental.prefetch_to_device("/gpu:0", tf.data.AUTOTUNE))
    )
    return ds


@register_task
@requires_gpu
def _main(_):
    """
    1. The parameters were picked out of the blue, this script's purpose is to demonstrate APIs.
    2. The script was specifically designed to run in NVIDIA's TensorFlow container with tag 22.12-tf2-py3,
    more about it here https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow.
    """
    logging.info(f"{tf.config.list_physical_devices() = }")

    experiment_name = FLAGS.experiment
    logs_dir = FLAGS.logs_dir
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir
    num_attention_heads = FLAGS.num_attention_heads
    num_stacks = FLAGS.num_stacks
    hidden_layer_size = FLAGS.hidden_size
    prng_seed = FLAGS.prng_seed
    initial_learning_rate = FLAGS.initial_learning_rate
    decay_alpha = FLAGS.decay_alpha
    log_frequency = FLAGS.log_frequency

    set_random_seed(prng_seed)

    if experiment_name == "electricity":
        experiment = tft.experiments.electricity_experiment
        map_fn = electricity_map_fn

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

    model = tft.make_tft_model(
        experiment,  # noqa
        # Those were picked randomly tbh.
        num_attention_heads=num_attention_heads,
        hidden_layer_size=hidden_layer_size,
        num_stacks=num_stacks,
    )
    model.compile(
        optimizer=make_optimizer(
            # Also picked randomly lol.
            CosineDecay(
                initial_learning_rate,
                int(steps_per_epoch * epochs),
                alpha=decay_alpha,
            ),
        ),
        # We can't use XLA and CuDNN at the same time.
        jit_compile=False,
    )
    model_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    callbacks = [
        TensorBoard(
            f"{logs_dir}/{experiment_name}/{model_tag}/tensorboard_logs/",
            update_freq=log_frequency,
            # Graph is pretty useless, unless debugging NaN's.
            write_graph=False,
            # Profile however, does provide some really helpfully details.
            # profile_batch=True,
            write_steps_per_second=True,
            profile_batch=True,
        ),
        TerminateOnNaN(),
    ]
    model.fit(
        training_split,
        epochs=epochs,
        validation_data=validation_split,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )

    with tf.device("/cpu:0"):
        # To avoid saving optimizer, we must use keras.
        model.save_weights(f"{logs_dir}/{experiment_name}/{model_tag}/weights.keras")


def main():
    run("tft_main")
