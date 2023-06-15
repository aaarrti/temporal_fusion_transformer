from __future__ import annotations

from typing import Dict, Tuple
import functools
from absl import flags, app
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
from keras.utils.tf_utils import can_jit_compile, set_random_seed
from keras.api.keras.experimental import CosineDecay
from keras.callbacks import TensorBoard, TerminateOnNaN, BackupAndRestore
from temporal_fusion_transformer import setup_logging, make_tft_model, make_gpu_strategy
from temporal_fusion_transformer.experiments import (
    electricity_experiment,
    favorita_experiment,
)

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
    experiment = None
    map_fn = None

    if experiment_name == "electricity":
        experiment = electricity_experiment
        map_fn = electricity_map_fn
    if experiment_name == "favorita":
        experiment = favorita_experiment
        map_fn = favorita_map_fn

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epoch
    data_dir = FLAGS.data_dir

    steps_per_epoch = num_electricity_samples // batch_size
    val_steps = num_val_samples // batch_size

    if can_jit_compile(True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.config.optimizer.set_jit("autoclustering")

    map_fn = functools.partial(
        map_fn, dtype=tf.keras.mixed_precision.global_policy().compute_dtype
    )
    strategy = make_gpu_strategy()

    def hyper_model(hp: kt.HyperParameters) -> kt.HyperModel:
        num_attention_heads = hp.Int("num_attention_heads", 1, 12)
        hidden_layer_size = hp.Int("hidden_layer_size", 8, 512)
        num_stacks = hp.Int("num_stacks", 1, 6)
        init_lr = hp.Float("learning_rate", 5e-3, 1e-1)
        decay_steps = hp.Int(
            "decay_steps",
            int(steps_per_epoch * epochs) // 2,
            int(steps_per_epoch * epochs * 2),
        )
        decay_alpha = hp.Float("decay_alpha", 0.02, 0.1)
        do_clip_norm = hp.Boolean("do_clip_norm", False)
        with hp.conditional_scope("do_clip_norm", [True]):
            if do_clip_norm:
                clip_norm_value = hp.Float("clip_norm_value", 0.01, 2)
            else:
                clip_norm_value = None

        model = make_tft_model(
            experiment,
            use_cudnn_lstm=True,
            num_attention_heads=num_attention_heads,
            hidden_layer_size=hidden_layer_size,
            num_stacks=num_stacks,
        )
        model.compile(
            optimizer=Adam(
                CosineDecay(
                    init_lr,
                    decay_steps,
                    alpha=decay_alpha,
                ),
                clipnorm=clip_norm_value,
            )
        )

        return model

    with strategy.scope():
        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                [f"{data_dir}/{experiment_name}/train/{i}" for i in range(19)]
            )
            .flat_map(
                lambda i: tf.data.Dataset.load(i, element_spec=experiment.element_spec)
            )
            .rebatch(batch_size, True)
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
            .rebatch(batch_size, True)
            .map(map_fn, tf.data.AUTOTUNE)
            .shuffle(32, PRNG_SEED, True)
            .cache()
            .repeat(epochs)
            .prefetch(tf.data.AUTOTUNE)
        )

        tuner = kt.Hyperband(
            hyper_model,
            max_epochs=epochs * 2,
            factor=2,
            directory=f"{data_dir}/{experiment_name}/keras_tuner",
            seed=PRNG_SEED,
            distribution_strategy=strategy,
            objective=kt.Objective("quantile_rmse", direction="min"),
            overwrite=True,
            project_name=f"tft_hyperband_{experiment_name}",
        )

        tuner.search(
            train_ds,
            epochs=epochs,
            validation_data=validation_ds,
            callbacks=[
                TensorBoard(
                    f"{data_dir}/{experiment_name}/tensorboard_logs",
                    update_freq=50,
                ),
                TerminateOnNaN(),
                BackupAndRestore(f"{data_dir}/{experiment_name}/checkpoints"),
            ],
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
        )


if __name__ == "__main__":
    app.run(main)
