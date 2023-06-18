from __future__ import annotations

import logging
from typing import Dict, Tuple, List
import functools
from absl import flags, app
import tensorflow as tf
from keras.utils.tf_utils import set_random_seed
from keras.api.keras.experimental import CosineDecay
from keras.callbacks import TensorBoard, TerminateOnNaN
from temporal_fusion_transformer import setup_logging, make_tft_model
from temporal_fusion_transformer import experiments
from temporal_fusion_transformer.src.utils import can_jit_compile

minor_tf_api_version = int(tf.__version__.split(".")[1])
if minor_tf_api_version >= 11:
    from keras.optimizers.adam import Adam
else:
    from keras.optimizers.optimizer_experimental.adam import Adam  # noqa

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "experiment",
    enum_values=["electricity", "favorita"],
    required=True,
    help="Name of the experiment",
    default=None,
)
flags.DEFINE_integer("batch_size", default=64, help="Training batch size")
flags.DEFINE_integer("epochs", default=1, help="Number of training epochs")
flags.DEFINE_string("data_dir", help="Data directory", default="gs://tf2_tft_v2/data")
flags.DEFINE_string("logs_dir", help=None, default="gs://tf2_tft_v2/logs")

PRNG_SEED = 42
set_random_seed(PRNG_SEED)
setup_logging()
num_electricity_samples = 1853057
num_val_samples = 204057

try:
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import nvidia.dali.plugin.tf as dali_tf

    is_dali_installed = True
except ModuleNotFoundError:
    logging.warning("DALI not installed, falling back to TF dataset")
    is_dali_installed = False


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
    logs_dir = FLAGS.logs_dir

    if experiment_name == "electricity":
        experiment = experiments.electricity_experiment
        map_fn = electricity_map_fn
    if experiment_name == "favorita":
        experiment = experiments.favorita_experiment
        map_fn = favorita_map_fn

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir

    steps_per_epoch = num_electricity_samples // batch_size
    val_steps = num_val_samples // batch_size

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if can_jit_compile(True):
        tf.config.optimizer.set_jit("autoclustering")

    map_fn = functools.partial(
        map_fn, dtype=tf.keras.mixed_precision.global_policy().compute_dtype  # noqa
    )

    def make_dataset(file_names: List[str]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_tensor_slices(file_names)
            .flat_map(
                lambda i: tf.data.Dataset.load(i, element_spec=experiment.element_spec)
            )
            # rebatch was added only in 2.12
            .unbatch()
            .batch(batch_size, True)
            .map(map_fn, tf.data.AUTOTUNE)
            .shuffle(32, PRNG_SEED, True)
            .cache()
            .repeat(epochs)
            .prefetch(tf.data.AUTOTUNE)
        )

    train_ds = make_dataset(
        [f"{data_dir}/{experiment_name}/train/{i}" for i in range(19)]
    )
    validation_ds = make_dataset(
        [f"{data_dir}/{experiment_name}/validation/{i}" for i in range(3)]
    )

    model = make_tft_model(
        experiment,  # noqa
        # Unrolling makes it worse, CuDNN is the GOAT.
        use_cudnn_lstm=True,
        # Those were picked randomly tbh.
        num_attention_heads=12,
        hidden_layer_size=180,
        num_stacks=4,
    )
    model.compile(
        optimizer=Adam(
            # Also picked randomly lol.
            CosineDecay(
                5e-3,
                int(steps_per_epoch * epochs),
                alpha=0.02,
            ),
            jit_compile=can_jit_compile(True),
        ),
        # Surprisingly, here XLA only slows everything down like 10+ times.
        jit_compile=False,
    )
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=validation_ds,
        callbacks=[
            TensorBoard(
                f"{logs_dir}/{experiment_name}/tensorboard_logs",
                update_freq=50,
                # Graph is pretty useless, unless debugging NaN's.
                write_graph=False,
                # Profile however, does provide some really helpfully details.
                profile_batch=True,
                write_steps_per_second=True
            ),
            TerminateOnNaN(),
            # No need really, unless running super large scale training.
            # BackupAndRestore(f"{logs_dir}/{experiment_name}/checkpoints"),
        ],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )

    with tf.device("cpu"):
        model.save_weights(f"{logs_dir}/{experiment_name}/weights_v1")


if __name__ == "__main__":
    app.run(main)
