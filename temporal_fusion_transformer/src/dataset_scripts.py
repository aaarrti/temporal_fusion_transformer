from __future__ import annotations

import functools
import logging
from datetime import datetime
from typing import Generator, Literal, Tuple
import platform

import jax
import optax
import tensorflow as tf
from absl_extra import flax_utils
from jax import numpy as jnp

from temporal_fusion_transformer.src.config_dict import ConfigDict
from temporal_fusion_transformer.src.datasets.electricity import Electricity
from temporal_fusion_transformer.src.datasets.favorita import Favorita
from temporal_fusion_transformer.src.quantile_loss import make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_model import InputStruct, make_input_struct_from_config, make_tft_model
from temporal_fusion_transformer.src.training import (
    MetricContainer,
    TrainStateContainer,
    multi_device_train_step,
    multi_device_validation_step,
    single_device_train_step,
    single_device_validation_step,
    make_training_hooks,
)

_experiment_factories = {
    "electricity": Electricity,
    "favorita": Favorita,
}
_EXPECTED_NUM_DEVICES = 8


def make_dataset(
    data_dir: str,
    experiment_name: Literal["electricity", "favorita", "volatility", "traffic"],
    **kwargs,
):
    """

    Parameters
    ----------
    data_dir:
        Path to directory with raw CSVs.
    experiment_name

    Returns
    -------

    """

    data_dir = f"{data_dir}/{experiment_name}"
    experiment = _experiment_factories[experiment_name](**kwargs)
    (train_ds, val_ds, test_ds), feature_space = experiment.make_dataset(data_dir)
    train_ds.save(f"{data_dir}/training")
    val_ds.save(f"{data_dir}/validation")
    test_ds.save(f"{data_dir}/test")
    feature_space.save(f"{data_dir}/features_space.keras")


def train_on_single_device(
    *,
    data_dir: str,
    batch_size: int,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDict,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str = "models",
):
    """

    Parameters
    ----------
    data_dir
    batch_size
    epochs
    experiment_name
    config
    mixed_precision:
        If set to True, will use (b)float16 for computations.
    jit_module:
        If set to True, will nn.jit flax module.
    save_path:
        Prefix of directory, in which models weights must be saved.

    Returns
    -------

    """

    compute_dtype = jnp.float16 if mixed_precision else jnp.float32

    def load_dataset(split: str) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{experiment_name}/{split}")
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(config.shuffle_buffer_size, seed=config.prng_seed)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    def generate_dataset(ds: tf.data.Dataset) -> Generator[Tuple[InputStruct, jnp.ndarray], None, None]:
        for x, y in ds.as_numpy_iterator():
            x = make_input_struct_from_config(x, config.fixed_params)
            yield x.cast_inexact(compute_dtype), jnp.asarray(y, compute_dtype)

    tag = make_timestamp_tag()
    log_dir = f"tensorboard/{experiment_name}/{tag}"
    logging.info(f"Writing tensorboard logs to {log_dir}")

    training_ds = load_dataset("training")
    validation_ds = load_dataset("validation")

    if platform.system().lower() == "darwin":
        logging.warning("Running on MacOS, will use 5 training and 2 validation batches for testing purposes.")
        training_ds = training_ds.take(5)
        validation_ds = validation_ds.take(2)

    num_training_steps = int(training_ds.cardinality())
    first_x = training_ds.as_numpy_iterator().next()[0]

    # --------------------------------------------------

    # We create 2 module, to avoid re-compilation. We use 1 set of parameters for both.
    training_model = make_tft_model(config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    # model_str = training_model.tabulate(params_key, first_x)
    # logging.info(f"Model -> {model_str}")
    params = training_model.init(params_key, first_x)["params"]

    learning_rate = config.optimizer.learning_rate
    if config.optimizer.decay_steps != 0:
        decay_steps = num_training_steps * epochs * config.optimizer.decay_steps
        learning_rate = optax.cosine_decay_schedule(learning_rate, decay_steps, config.optimizer.decay_alpha)
    tx = optax.adam(learning_rate)
    if config.optimizer.clipnorm != 0:
        tx = optax.chain(optax.adaptive_grad_clip(config.optimizer.clipnorm), tx)

    if config.optimizer.ema != 0:
        tx = optax.chain(tx, optax.ema(config.optimizer.ema))

    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles)
    state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=training_model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
        # 1 device is always a GPU
        compute_dtype=jnp.float16 if mixed_precision else jnp.float32,
    )

    hooks = make_training_hooks(
        num_training_steps,
        epochs,
        log_frequency=2,
        logdir=log_dir,
    )

    (training_metrics, validation_metrics), params = flax_utils.fit_single_device(
        training_state=state,
        training_dataset_factory=functools.partial(generate_dataset, training_ds),
        validation_dataset_factory=functools.partial(generate_dataset, validation_ds),
        metrics_container_type=MetricContainer,
        training_step_func=single_device_train_step,
        validation_step_func=single_device_validation_step,
        epochs=epochs,
        hooks=hooks,
        num_training_steps=num_training_steps,
    )
    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")
    flax_utils.save_as_msgpack(params, f"{save_path}/{experiment_name}/{tag}/model.msgpack")


def train_on_multiple_devices(
    *,
    data_dir: str,
    batch_size: int,
    epochs: int,
    log_frequency,
    log_dir,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDict,
):
    """
    This script is designed to run on TPU. It expects you to initialize TPU system for jax beforehand.
    If you want to run it locally for debugging purposes,
    set environment variable `XLA_FLAGS="--xla_force_host_platform_device_count=8"`
    """
    log_dir = f"{log_dir}/{experiment_name}"

    def load_dataset(split: str) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{experiment_name}/{split}")
            .batch(batch_size * _EXPECTED_NUM_DEVICES, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(config.shuffle_buffer_size, seed=config.prng_seed, reshuffle_each_iteration=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_dataset("training")
    validation_ds = load_dataset("validation")

    num_training_steps = int(training_ds.cardinality())
    x = training_ds.as_numpy_iterator().next()[0]

    # --------------------------------------------------

    # We create 2 module, to avoid re-compilation. We use 1 set of parameters for both.
    training_model = make_tft_model(config)
    validation_model = make_tft_model(config)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    training_model.tabulate(params_key, x)
    params = training_model.init(params_key, x)["params"]

    # TODO: don't replicate large kernel.

    decay_steps = num_training_steps * epochs * config.optimizer.decay_steps
    lr = optax.cosine_decay_schedule(config.optimizer.learning_rate, decay_steps, config.optimizer.decay_alpha)
    tx = optax.adam(lr)

    loss_fn = make_quantile_loss_fn(config.quantiles)
    state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=training_model.apply,
        validation_fn=validation_model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
    )

    hooks = flax_utils.make_training_hooks(
        num_training_steps,
        epochs,
        log_frequency=log_frequency,
        logdir=log_dir,
    )

    (training_metrics, validation_metrics), params = flax_utils.fit(
        training_state=state,
        training_dataset_factory=lambda: training_ds.as_numpy_iterator(),
        validation_dataset_factory=lambda: validation_ds.as_numpy_iterator(),
        prefetch_buffer_size=0,
        metrics_container_type=MetricContainer,
        training_step_func=train_step,
        validation_step_func=validation_step,
        epochs=epochs,
        hooks=hooks,
    )
    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")
    flax_utils.save_as_msgpack(f"models/{experiment_name}/model.msgpack")


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")
