from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Literal, Tuple

import keras_core as keras
from keras_core import callbacks, optimizers
from ml_collections import ConfigDict
from toolz import dicttoolz

from temporal_fusion_transformer.src.modeling import TemporalFusionTransformer
from temporal_fusion_transformer.src.quantile_loss import QuantileLoss
from temporal_fusion_transformer.src.quantile_metrics import make_quantile_error_metrics
from temporal_fusion_transformer.src.config_dict import Config, OptimizerConfig

if TYPE_CHECKING:
    import tensorflow as tf

log = logging.getLogger(__name__)


def train_model(
    *,
    dataset: tuple[tf.data.Dataset, tf.data.Dataset],
    config: Config,
    epochs: int = 1,
    training_callbacks="auto",
    save_filename: str = "model.keras",
    verbose="auto",
    **kwargs,
):
    model = TemporalFusionTransformer(
        **config.model.to_dict(), **config.data.to_dict(), num_quantiles=len(config.quantiles)
    )

    num_train_steps = int(dataset[0].cardinality())
    quantiles = config.quantiles

    allowed_compile_kwargs = inspect.signature(model.compile).parameters
    allowed_fit_kwargs = inspect.signature(model.fit).parameters
    compile_kwargs = dicttoolz.keyfilter(lambda k: k in allowed_compile_kwargs, kwargs)
    fit_kwargs = dicttoolz.keyfilter(lambda k: k in allowed_fit_kwargs, kwargs)

    model.compile(
        loss=QuantileLoss(quantiles),
        optimizer=make_optimizer(config.optimizer, num_train_steps),
        metrics=make_quantile_error_metrics(quantiles),
        **compile_kwargs,
    )

    x = dataset[0].as_numpy_iterator().next()[0]
    verify_declared_number_of_features_matches(config.data, x)
    # if verbose == "auto":
    #   model(x)
    #   keras.utils.plot_model(
    #       model,
    #       show_layer_names=True,
    #       expand_nested=True,
    #       show_layer_activations=True,
    #       show_shapes=True,
    #       # show_dtype=True,
    #   )
    #   model.summary(expand_nested=True)

    if training_callbacks == "auto":
        training_callbacks = default_callbacks()

    model.fit(
        dataset[0],
        validation_data=dataset[1],
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=verbose,
        **fit_kwargs,
    )

    if save_filename is not None:
        model.save_weights(save_filename)
        return None
    else:
        return model


def default_callbacks():
    return [
        callbacks.TerminateOnNaN(),
        callbacks.EarlyStopping(
            min_delta=0.01, patience=1, start_from_epoch=1, restore_best_weights=True, verbose=1
        ),
        callbacks.TensorBoard(
            write_graph=False,
            log_dir="tensorboard",
            update_freq=100,
        ),
        callbacks.ModelCheckpoint(
            filepath="checkpoints",
            save_freq=1000,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def make_optimizer(config: OptimizerConfig, num_train_steps: int):
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

    if config.decay_steps != 0:
        lr = optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=int(config.decay_steps * num_train_steps),
            alpha=config.alpha,
        )
    else:
        lr = config.learning_rate

    return optimizers.Lion(
        learning_rate=lr, use_ema=config.use_ema, clipnorm=clipnorm, weight_decay=weight_decay
    )


def load_dataset(
    data_dir: str,
    batch_size: int,
    encoder_steps: int,
    prng_seed: int = 33,
    shuffle_buffer_size: int = 1024,
    dtype="float32",
    element_spec: Tuple[tf.TensorSpec, tf.TensorSpec] | None = None,
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
    encoder_steps:
        Number of time steps to consider as past. Those steps will be discarded from y_batch.
    element_spec

    Returns
    -------

    """
    import tensorflow as tf

    tf_dtype = tf.dtypes.as_dtype(dtype)

    def downcast_input(x, y):
        return tf.cast(x, tf_dtype), tf.cast(y, tf_dtype)

    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(
                f"{data_dir}/{split}", compression="GZIP", element_spec=element_spec
            )
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, seed=prng_seed, reshuffle_each_iteration=True)
            .map(downcast_input)
            .map(lambda x, y: (x, y[:, encoder_steps:]))
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")
    return training_ds, validation_ds


def verify_declared_number_of_features_matches(config: ConfigDict, x_batch):
    declared_num_features = (
        len(config.input_static_idx)
        + len(config.input_known_real_idx)
        + len(config.input_known_categorical_idx)
        + len(config.input_observed_idx)
    )
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
