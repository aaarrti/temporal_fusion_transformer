from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import TYPE_CHECKING, Literal

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks, optimizers
from ml_collections import ConfigDict
from toolz import dicttoolz

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.modeling.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.quantile_loss import PinballLoss, QuantilePinballLoss
from temporal_fusion_transformer.src.utils import count_inputs

log = logging.getLogger(__name__)


class TerminateOnNaN(callbacks.TerminateOnNaN):
    def __init__(self):
        super().__init__()
        self.weights = None

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        if self.model.stop_training and self.weights is not None:
            log.warning("Restoring weight from previous epoch.")
            self.model.weights = self.weights

    def on_epoch_end(self, epoch, logs=None):
        self.weights = self.model.weights


def train_model_from_config(
    *,
    dataset: tuple[tf.data.Dataset, tf.data.Dataset],
    config: Config,
    callbacks=None,
) -> keras.Model:
    model = TemporalFusionTransformer.from_dataclass_config(config)

    num_train_steps = int(dataset[0].cardinality())
    quantiles = config.quantiles

    model.compile(
        loss=QuantilePinballLoss(quantiles),
        optimizer=make_optimizer(config, num_train_steps),
        jit_compile=False,
    )

    x = dataset[0].as_numpy_iterator().next()[0]
    verify_declared_number_of_features_matches(config, x)

    model.fit(
        dataset[0],
        validation_data=dataset[1],
        epochs=config.epochs,
        callbacks=callbacks,
    )
    return model


def make_optimizer(config: Config, num_train_steps: int):
    """

    config must contain following fields:

    - initial_learning_rate
    - decay_steps
    - alpha
    - use_ema
    - clipnorm
    - weight_decay
    - num_train_steps

    """
    if config.decay_steps is not None:
        lr = optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=int(config.decay_steps * num_train_steps),
            alpha=config.decay_alpha,
        )
    else:
        lr = config.learning_rate

    return optimizers.Adam(
        learning_rate=lr,
        use_ema=config.use_ema,
        clipnorm=config.clipnorm,
        weight_decay=config.weight_decay,
    )


def load_dataset(
    data_dir: str,
    batch_size: int,
    encoder_steps: int,
    shuffle_buffer_size: int = 1024,
    dtype="float32",
    element_spec: tuple[tf.TensorSpec, tf.TensorSpec] | None = None,
    compression: Literal["GZIP"] | None = "GZIP",
    drop_remainder: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """

    Parameters
    ----------
    data_dir
    batch_size
    shuffle_buffer_size:
        If set to None, will do a full-reshuffle.
    dtype
    encoder_steps:
        Number of time steps to consider as past. Those steps will be discarded from y_batch.
    element_spec
    compression
    drop_remainder

    Returns
    -------

    """

    tf_dtype = tf.dtypes.as_dtype(dtype)

    def downcast_input(x, y):
        return tf.cast(x, tf_dtype), tf.cast(y, tf_dtype)

    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        load_kwargs = {}
        if compression is not None:
            load_kwargs["compression"] = compression

        return (
            tf.data.Dataset.load(f"{data_dir}/{split}", element_spec=element_spec, **load_kwargs)
            .batch(batch_size, drop_remainder=drop_remainder, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
            .map(downcast_input)
            .map(lambda x, y: (x, y[:, encoder_steps:]))
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")
    return training_ds, validation_ds


def load_dataset_from_config(
    data_dir: str, config: Config
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    num_inputs = count_inputs(config)

    return load_dataset(
        data_dir=data_dir,
        batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        drop_remainder=config.drop_remainder,
        compression=config.compression,
        encoder_steps=config.encoder_steps,
        element_spec=(
            tf.TensorSpec(
                shape=(config.total_time_steps, num_inputs), dtype=tf.float32, name="inputs"
            ),
            tf.TensorSpec(
                shape=(config.total_time_steps, config.num_outputs),
                dtype=tf.float32,
                name="outputs",
            ),
        ),
    )


def verify_declared_number_of_features_matches(config: ConfigDict, x_batch: np.ndarray):
    declared_num_features = count_inputs(config)
    num_features = x_batch.shape[-1]

    if num_features != declared_num_features:
        unknown_indexes = sorted(
            list(
                set(
                    config.input_static_idx
                    + config.input_known_real_idx
                    + config.input_known_categorical_idx
                    + config.input_observed_idx
                ).symmetric_difference(range(num_features))
            )
        )
        if num_features != declared_num_features:
            raise RuntimeError(
                f"Declared number of features ({declared_num_features}) does not match with the one seen in input ({num_features})"
                f"could not indentify inputs at {unknown_indexes}"
            )
        else:
            log.info("Num features check -> OK")
