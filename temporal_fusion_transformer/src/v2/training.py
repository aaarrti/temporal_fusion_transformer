from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from keras_core import ops
from keras_core import optimizers, metrics, callbacks
from keras_core.mixed_precision import global_policy

from temporal_fusion_transformer.src.v2.modeling import TemporalFusionTransformer
from temporal_fusion_transformer.src.v2.loss_fn import QuantilePinballLoss
from ml_collections import ConfigDict

if TYPE_CHECKING:
    import tensorflow as tf
    from keras_core.src.backend.jax.trainer import JAXTrainer


def train(
    *,
    data: tuple[tf.data.Dataset, tf.data.Dataset],
    config: ConfigDict,
    epochs: int = 1,
    steps_per_execution: int = 1,
    training_callbacks="auto",
    save_filename: str = "model.keras",
):
    model = TemporalFusionTransformer.from_config(config.model.to_dict())

    num_train_steps = int(data[0].cardinality())

    model.compile(
        loss=QuantilePinballLoss(dtype=global_policy().compute_dtype, quantiles=config.model.quantiles),
        metrics=make_metrics(),
        optimizer=make_optimizer(config.optimizer, num_train_steps),
        steps_per_execution=steps_per_execution,
    )

    if training_callbacks == "auto":
        training_callbacks = default_callbacks()

    model.fit(data[0], validation_data=data[1], epochs=epochs, callbacks=training_callbacks, verbose=1)
    model.save_weights(save_filename)


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
    ]


def make_metrics():
    return []


def make_optimizer(config: ConfigDict, num_train_steps: int):
    return _make_optimizer(
        num_train_steps=num_train_steps,
        **config.to_dict(),
    )


def _make_optimizer(
    *,
    initial_learning_rate: float,
    decay_steps: float,
    alpha: float,
    use_ema: bool,
    num_train_steps: int,
    clipnorm: float | None,
    weight_decay: float | None,
) -> optimizers.Optimizer:
    """

    Parameters
    ----------
    initial_learning_rate
    decay_steps
    alpha
    use_ema
    clipnorm
    weight_decay
    num_train_steps

    Returns
    -------

    """

    if clipnorm == 0:
        clipnorm = None
    if weight_decay == 0:
        weight_decay = None

    lr = optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=int(decay_steps * num_train_steps), alpha=alpha
    )

    return optimizers.Lion(learning_rate=lr, use_ema=use_ema, clipnorm=clipnorm, weight_decay=weight_decay)


def load_dataset(
    data_dir: str,
    batch_size: int,
    prng_seed: int,
    num_encoder_steps: int,
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
