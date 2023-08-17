from __future__ import annotations

import functools
from typing import Callable, Literal

import jax
from absl import logging
from absl_extra import flax_utils
from absl_extra.typing_utils import ParamSpec
from flax.core.frozen_dict import FrozenDict
from flax.training.dynamic_scale import DynamicScale
from jax import numpy as jnp
from jax.sharding import NamedSharding

from temporal_fusion_transformer.src.config_dict import ConfigDict
from temporal_fusion_transformer.src.quantile_loss import make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.training_lib import (
    MetricContainer,
    TrainStateContainer,
    load_dataset,
    make_dataset_generator_func,
    make_optimizer,
    make_training_hooks,
    multi_device_train_step,
    multi_device_validation_step,
    single_device_train_step,
    single_device_validation_step,
)
from temporal_fusion_transformer.src.utils import make_timestamp_tag

P = ParamSpec("P")


def train_experiment_on_single_device(
    *,
    data_dir: str,
    batch_size: int,
    config: ConfigDict,
    experiment_name: Literal["electricity", "favorite"],
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    stop_early: bool = False,
    tensorboard_log_dir: str = "tensorboard",
    checkpoint_dir: str | None = "checkpoints",
    log_frequency: int = 10,
):
    train_on_single_device(
        data_dir=f"{data_dir}/{experiment_name}",
        batch_size=batch_size,
        config=config,
        epochs=epochs,
        mixed_precision=mixed_precision,
        jit_module=jit_module,
        save_path=save_path,
        stop_early=stop_early,
        tensorboard_log_dir=f"{tensorboard_log_dir}/{experiment_name}",
        checkpoint_dir=checkpoint_dir,
        log_frequency=log_frequency,
    )


def train_on_single_device(
    *,
    data_dir: str,
    batch_size: int,
    config: ConfigDict,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = "model.msgpack",
    stop_early: bool = False,
    tensorboard_log_dir: str = "tensorboard",
    checkpoint_dir: str | None = "checkpoints",
    profile: bool = False,
    log_frequency: int = 10,
    verbose: bool = True,
):
    compute_dtype = jnp.float16 if mixed_precision else jnp.float32

    tag = make_timestamp_tag()

    tensorboard_log_dir = f"{tensorboard_log_dir}/{tag}"

    logging.info(f"Writing tensorboard logs to {tensorboard_log_dir}")

    training_dataset, validation_dataset = load_dataset(
        data_dir, batch_size, config.shuffle_buffer_size, config.prng_seed
    )

    generator_func = make_dataset_generator_func(compute_dtype, config.fixed_params)

    num_training_steps = int(training_dataset.cardinality())
    first_x = next(generator_func(training_dataset))[0]

    # --------------------------------------------------

    model = TemporalFusionTransformer.from_config_dict(config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    params = model.init(params_key, first_x)["params"]

    tx = make_optimizer(config.optimizer, num_training_steps, epochs)

    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles, dtype=compute_dtype)

    if mixed_precision:
        # TODO allow low scale args to be passed
        dynamic_scale = DynamicScale()
    else:
        dynamic_scale = None

    state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
        dynamic_scale=dynamic_scale,
    )

    hooks = make_training_hooks(
        num_training_steps,
        epochs,
        log_frequency=log_frequency,
        logdir=tensorboard_log_dir,
        add_early_stopping=stop_early,
        profile=profile,
    )

    (training_metrics, validation_metrics), params = flax_utils.fit_single_device(
        training_state=state,
        training_dataset_factory=functools.partial(generator_func, training_dataset),
        validation_dataset_factory=functools.partial(generator_func, validation_dataset),
        metrics_container_type=MetricContainer,
        training_step_func=single_device_train_step,
        validation_step_func=single_device_validation_step,
        epochs=epochs,
        hooks=hooks,
        num_training_steps=num_training_steps,
        verbose=verbose,
    )
    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")

    if save_path is not None:
        flax_utils.save_as_msgpack(params, save_path)


def train_on_multiple_devices(
    *,
    data_dir: str,
    batch_size: int,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDict,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str | None = None,
    tabulate_model: bool = False,
    data_sharding: NamedSharding | None = None,
    params_replication: Callable[[FrozenDict], FrozenDict] | None = None,
    device_type: Literal["gpu", "tpu"] = "tpu",
):
    """

    This script is designed to run on TPU. It expects you to initialize TPU system for jax beforehand.
    If you want to run it locally for debugging purposes,
    set environment variable `XLA_FLAGS="--xla_force_host_platform_device_count=8"`
    Probably, it can also work in multi-GPU environment, but it was not tested.

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
    tabulate_model:
        Log model structure yes/no. It can be useful to see which modules take the most memory,
        in order to adapt sharding/replication strategy.

    data_sharding:
        NamesSharding, in case you want more fine-grained control on how data is sharded across devices.
        Applies only to distributed training.
    params_replication:
        NamedSharding, in case you want more fine-grained control on how params are replicated across replicas,
        e.g., you might want to shard large kernel instead of replicating them (or both).
    device_type:
        GPU or TPU.

    """

    if mixed_precision:
        if device_type == "gpu":
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.bfloat16
    else:
        compute_dtype = jnp.float32

    num_devices = jax.device_count()

    tag = make_timestamp_tag()
    log_dir = f"tensorboard/{experiment_name}/{tag}"
    logging.info(f"Writing tensorboard logs to {log_dir}")

    training_dataset, validation_dataset = load_dataset(
        data_dir, experiment_name, batch_size * num_devices, config.shuffle_buffer_size, config.prng_seed
    )

    generator_func = make_dataset_generator_func(compute_dtype, config.fixed_params)

    num_training_steps = int(training_dataset.cardinality())
    first_x = next(generator_func(training_dataset))[0]

    model = make_tft_model(config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    if tabulate_model:
        model_str = model.tabulate(params_key, first_x)
        logging.info(model_str)

    params = model.init(params_key, first_x)["params"]

    if params_replication:
        params = params_replication(params)

    tx = make_optimizer(config.optimizer, num_training_steps, epochs)

    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles, dtype=compute_dtype)

    if mixed_precision and device_type == "gpu":
        loss_scale = DynamicScale()
    else:
        loss_scale = None

    state = TrainStateContainer.create(
        params=params, tx=tx, apply_fn=model.apply, dropout_key=dropout_key, loss_fn=loss_fn, loss_scale=loss_scale
    )

    hooks = make_training_hooks(
        num_training_steps,
        epochs,
        log_frequency=2,
        logdir=log_dir,
    )

    prefetch_buffer_size = 2 if device_type == "gpu" else 0

    (training_metrics, validation_metrics), params = flax_utils.fit_multi_device(
        training_state=state,
        training_dataset_factory=functools.partial(generator_func, training_dataset),
        validation_dataset_factory=functools.partial(generator_func, validation_dataset),
        metrics_container_type=MetricContainer,
        training_step_func=multi_device_train_step,
        validation_step_func=multi_device_validation_step,
        epochs=epochs,
        hooks=hooks,
        num_training_steps=num_training_steps,
        data_sharding=data_sharding,
        prefetch_buffer_size=prefetch_buffer_size,
    )
    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")

    if save_path is None:
        save_path = f"{save_path}/{experiment_name}/{tag}/model.msgpack"

    flax_utils.save_as_msgpack(params, save_path)