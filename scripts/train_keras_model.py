from __future__ import annotations

from typing import Dict, Tuple
import functools
from absl import flags, app
import tensorflow as tf
from keras.utils.tf_utils import set_random_seed
from keras.api.keras.experimental import CosineDecay
from keras.callbacks import TensorBoard, TerminateOnNaN, BackupAndRestore
from temporal_fusion_transformer import setup_logging, make_tft_model
from temporal_fusion_transformer.experiments import (
    electricity_experiment,
    favorita_experiment,
)
from temporal_fusion_transformer.utils import can_jit_compile

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita"],
    required=True,
    help="Name of the experiment",
    default=None,
)
flags.DEFINE_integer("batch_size", default=64, help="Training batch size")
flags.DEFINE_integer("epochs", default=10, help="Number of training epochs")
flags.DEFINE_string("data_dir", help="Data directory", default="gs://tf2_tft_v2/")
flags.DEFINE_string("persist_dir", help=None, default="logs")
flags.DEFINE_boolean("mixed_precision", default=True, help=None)
flags.DEFINE_boolean("use_xla", default=True, help=None)
flags.DEFINE_boolean("use_cudnn", default=False, help=None)


PRNG_SEED = 42
set_random_seed(PRNG_SEED)
setup_logging()
num_electricity_samples = 1853057
num_val_samples = 204057


def electricity_map_fn(
    arg: Dict[str, tf.Tensor], dtype
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    return (
        dict(
            static=arg["inputs_static"],
            known_real=tf.cast(arg["inputs_known_real"], dtype),
        ),
        tf.cast(arg["outputs"], dtype),
    )


def favorita_map_fn(
    arg: Dict[str, tf.Tensor], dtype
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    outputs = tf.cast(arg.pop("outputs"), dtype)
    return arg, outputs


def main(_):
    """The parameters were picked out of the blue, this script's purpose is to demonstrate APIs."""
    experiment_name = FLAGS.experiment
    persist_dir = FLAGS.persist_dir
    experiment = None
    map_fn = None
    use_xla = FLAGS.use_xla
    

    if experiment_name == "electricity":
        experiment = electricity_experiment
        map_fn = electricity_map_fn
    if experiment_name == "favorita":
        experiment = favorita_experiment
        map_fn = favorita_map_fn

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir

    steps_per_epoch = num_electricity_samples // batch_size
    val_steps = num_val_samples // batch_size

    if FLAGS.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
    if can_jit_compile() and use_xla:
        tf.config.optimizer.set_jit("autoclustering")

    map_fn = functools.partial(
        map_fn, dtype=tf.keras.mixed_precision.global_policy().compute_dtype
    )
    
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            [f"{data_dir}/{experiment_name}/train/{i}" for i in range(19)]
        )
        .flat_map(
            lambda i: tf.data.Dataset.load(i, element_spec=experiment.element_spec)
        )
        .unbatch()
        .batch(batch_size, True)
        .map(map_fn, tf.data.AUTOTUNE)
        .shuffle(32, PRNG_SEED, True)
        .cache()
        .repeat(epochs)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_ds = (
        tf.data.Dataset.from_tensor_slices(
            [f"{data_dir}/{experiment_name}/validation/{i}" for i in range(3)]
        )
        .flat_map(
            lambda i: tf.data.Dataset.load(i, element_spec=experiment.element_spec)
        )
        .unbatch()
        .batch(batch_size, True)
        .map(map_fn, tf.data.AUTOTUNE)
        .shuffle(32, PRNG_SEED, True)
        .cache()
        .repeat(epochs)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = make_tft_model(
        experiment,
        use_cudnn_lstm=FLAGS.use_xla,
        num_attention_heads=12,
        hidden_layer_size=180,
        num_stacks=4,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            CosineDecay(
                5e-3,
                int(steps_per_epoch * epochs),
                alpha=0.02,
            ),
            jit_compile=can_jit_compile(True)
        ),
        jit_compile=can_jit_compile(True) and use_xla,
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        callbacks=[
            TensorBoard(
                f"{persist_dir}/{experiment_name}/tensorboard_logs",
                update_freq=50,
                write_graph=False
            ),
            TerminateOnNaN(),
            BackupAndRestore(f"{persist_dir}/{experiment_name}/checkpoints"),
        ],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
    )

    with tf.device("cpu"):
        model.save_weights(f"{persist_dir}/{experiment_name}/weights_v1.keras")


if __name__ == "__main__":
    app.run(main)
