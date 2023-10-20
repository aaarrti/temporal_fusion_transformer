from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Literal

from keras_core import callbacks, distribution, optimizers
from ml_collections import ConfigDict

from temporal_fusion_transformer.src.loss_fn import QuantilePinballLoss
from temporal_fusion_transformer.src.metrics import make_quantile_error_metrics
from temporal_fusion_transformer.src.modeling import TemporalFusionTransformer

if TYPE_CHECKING:
    import tensorflow as tf
    from keras_core.src.backend.jax.trainer import JAXTrainer


class XlaGCCallback(callbacks.Callback):
    """See https://github.com/google/jax/issues/14882."""

    def __init__(self, interval=30):
        super().__init__()
        self.interval = interval

    def on_batch_end(self, batch, logs=None):
        if batch >= self.interval and batch % self.interval == 0:
            gc.collect()


def train_model(
    *,
    dataset: tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    epochs: int = 1,
    steps_per_execution: int = 1,
    training_callbacks="auto",
    save_filename: str = "model.keras",
    verbose="auto",
):
    model: JAXTrainer | TemporalFusionTransformer = TemporalFusionTransformer.from_config_dict(config)

    num_train_steps = int(dataset[0].cardinality())
    quantiles = config.quantiles

    model.compile(
        loss=QuantilePinballLoss(quantiles),
        optimizer=make_optimizer(config.model.optimizer, num_train_steps),
        steps_per_execution=steps_per_execution,
        metrics=make_quantile_error_metrics(quantiles),
    )

    if verbose == "auto":
        x = dataset[0].as_numpy_iterator().next()[0]
        model(x)
        model.summary(expand_nested=True)

    if training_callbacks == "auto":
        training_callbacks = default_callbacks()

    model.fit(dataset[0], validation_data=dataset[1], epochs=epochs, callbacks=training_callbacks, verbose=verbose)

    if save_filename is not None:
        model.save_weights(save_filename)
        return None
    else:
        return model


def default_callbacks():
    return [
        callbacks.TerminateOnNaN(),
        callbacks.EarlyStopping(min_delta=0.01, patience=1, start_from_epoch=1, restore_best_weights=True, verbose=1),
        callbacks.TensorBoard(
            write_graph=False,
            log_dir="tensorboard",
            update_freq=100,
        ),
        callbacks.ModelCheckpoint(
            filepath="checkpoints", save_freq=1000, save_best_only=True, save_weights_only=True, verbose=1
        ),
        XlaGCCallback(),
    ]


def make_optimizer(config: ConfigDict, num_train_steps: int):
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
    clipnorm = config.clipnorm
    weight_decay = config.weight_decay

    if clipnorm == 0:
        clipnorm = None
    if weight_decay == 0:
        weight_decay = None

    if config.decay_steps == 0:
        lr = optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=int(config.decay_steps * num_train_steps),
            alpha=config.alpha,
        )
    else:
        lr = config.learning_rate

    return optimizers.Lion(learning_rate=lr, use_ema=config.use_ema, clipnorm=clipnorm, weight_decay=weight_decay)


def load_dataset(
    data_dir: str,
    batch_size: int,
    num_encoder_steps: int,
    prng_seed: int = 33,
    shuffle_buffer_size: int = 1024,
    dtype="float32",
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """

    Parameters
    ----------
    data_dir
    batch_size
    shuffle_buffer_size:
        If set to None, will do a full-reshuffle.
    prng_seed
    dtype
    num_encoder_steps:
        Number of time steps to consider as past. Those steps will be discarded from y_batch.

    Returns
    -------

    """
    import tensorflow as tf

    tf_dtype = tf.dtypes.as_dtype(dtype)

    def downcast_input(x, y):
        return tf.cast(x, tf_dtype), tf.cast(y, tf_dtype)

    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{split}", compression="GZIP")
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, seed=prng_seed, reshuffle_each_iteration=True)
            .map(downcast_input)
            .map(lambda x, y: (x, y[:, num_encoder_steps:]))
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")
    return training_ds, validation_ds


def train_model_distributed(
    *,
    dataset: tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    epochs: int = 1,
    steps_per_execution: int = 1,
    training_callbacks="auto",
    save_filename: str = "model.keras",
    verbose="auto",
):
    """
    We support only data-parallel distributed training out of the box.
    """

    # Let Keras initialize it with defaults, since we are not doing any complicated partitioning/splitting/
    mesh = distribution.DataParallel()
    distribution.set_distribution(mesh)
