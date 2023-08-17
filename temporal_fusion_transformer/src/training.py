from __future__ import annotations

import functools
import logging
import platform
from datetime import datetime
from typing import Generator, Literal, Mapping, Protocol, Tuple, Callable

import jax
from jax.tree_util import Partial
from jax.sharding import NamedSharding
import optax
import tensorflow as tf
from absl_extra import flax_utils
from clu.metrics import Average
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass, field
from flax.training.dynamic_scale import DynamicScale
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from temporal_fusion_transformer.src.config_dict import ConfigDict, FixedParamsConfig, OptimizerConfig
from temporal_fusion_transformer.src.quantile_loss import QuantileLossFn, make_quantile_loss_fn
from temporal_fusion_transformer.src.tft_layers import ComputeDtype, InputStruct
from temporal_fusion_transformer.src.tft_model import make_input_struct_from_config, make_tft_model


class ApplyFunc(Protocol):
    def __call__(
        self,
        params: Mapping[str, FrozenDict],
        x: Float[Array, "batch time n"],
        training: bool = False,
        *,
        rngs: Mapping[str, PRNGKeyArray] | None = None,
    ) -> Float[Array, "batch time n"]:
        ...


@jaxtyped
@dataclass
class MetricContainer(flax_utils.AnnotationsCompatibleCollection):
    loss: Average.from_output("loss")


@jaxtyped
class TrainStateContainer(TrainState):
    apply_fn: ApplyFunc = field(pytree_node=False)
    loss_fn: QuantileLossFn = field(pytree_node=False)
    dropout_key: PRNGKeyArray
    loss_scale: DynamicScale = field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, dtype: ComputeDtype = jnp.float32, **kwargs) -> TrainStateContainer:
        if dtype == jnp.float16:
            loss_scale = DynamicScale(minimum_scale=jnp.finfo(jnp.float16).tiny)
        else:
            loss_scale = None

        return super().create(apply_fn=apply_fn, params=params, tx=tx, loss_scale=loss_scale, **kwargs)


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    log_frequency: int = 10,
) -> flax_utils.TrainingHooks:
    # TODO: add checkpoint
    # TODO: add early stopping
    hooks = flax_utils.make_training_hooks(
        num_training_steps=num_training_steps,
        epochs=epochs,
        write_metrics_frequency=log_frequency,
        report_progress_frequency=log_frequency,
        tensorboard_logdir=logdir,
    )
    return hooks


@jaxtyped
@jax.jit
def single_device_train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

    def loss_fn(params: FrozenDict) -> float:
        # pass training=True as positional args, since flax.nn.jit does not support kwargs.
        y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.loss_scale is not None:
        _, _, loss, grads = state.loss_scale.value_and_grad(loss_fn)(state.params)
    else:
        loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = MetricContainer.single_from_model_output(loss=loss)
    return state, metrics


@jaxtyped
@jax.jit
def single_device_validation_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y).mean()
    metrics = MetricContainer.single_from_model_output(loss=loss)
    return metrics


@jaxtyped
@functools.partial(jax.pmap, axis_name="i")
def multi_device_train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

    def loss_fn(params: FrozenDict) -> float:
        y = state.apply_fn({"params": params}, x_batch, training=True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.loss_fn is not None:
        _, _, loss, grads = state.loss_scale.value_and_grad(loss_fn, axis_name="i")(state.params)
    else:
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name="i")

    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jaxtyped
@functools.partial(jax.pmap, axis_name="i")
def multi_device_validation_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> MetricContainer:
    y = state.apply_fn({"params": state.params}, x_batch)
    loss = state.loss_fn(y_batch, y)
    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    return metrics


def load_dataset(
    data_dir: str,
    experiment_name: str,
    batch_size: int,
    shuffle_buffer_size: int,
    prng_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{experiment_name}/{split}", compression="GZIP")
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, seed=prng_seed)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")

    if platform.system().lower() == "darwin":
        logging.warning("Running on MacOS, will use 5 training and 2 validation batches for testing purposes.")
        training_ds = training_ds.take(5)
        validation_ds = validation_ds.take(2)

    return training_ds, validation_ds


def make_dataset_generator_func(
    compute_dtype: ComputeDtype, config: FixedParamsConfig
) -> Callable[[tf.data.Dataset], Generator[Tuple[InputStruct, jnp.ndarray], None, None]]:
    def generate_dataset(
        ds: tf.data.Dataset,
    ):
        for x, y in ds.as_numpy_iterator():
            x = make_input_struct_from_config(x, config, dtype=compute_dtype)
            yield x.cast_inexact(compute_dtype), jnp.asarray(y, compute_dtype)

    return generate_dataset


def make_optimizer(config: OptimizerConfig, num_training_steps: int, epochs: int) -> optax.GradientTransformation:
    learning_rate = config.learning_rate
    if config.decay_steps != 0:
        decay_steps = num_training_steps * epochs * config.decay_steps
        learning_rate = optax.cosine_decay_schedule(learning_rate, decay_steps, config.decay_alpha)
    tx = optax.adam(learning_rate)
    if config.clipnorm != 0:
        tx = optax.chain(optax.adaptive_grad_clip(config.clipnorm), tx)

    if config.ema != 0:
        tx = optax.chain(tx, optax.ema(config.ema))

    return tx


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
    Single device is always a GPU.

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

    tag = make_timestamp_tag()
    log_dir = f"tensorboard/{experiment_name}/{tag}"
    logging.info(f"Writing tensorboard logs to {log_dir}")

    training_dataset, validation_dataset = load_dataset(
        data_dir, experiment_name, batch_size, config.shuffle_buffer_size, config.prng_seed
    )

    generator_func = make_dataset_generator_func(compute_dtype, config.fixed_params)

    num_training_steps = int(training_dataset.cardinality())
    first_x = next(generator_func(training_dataset))[0]

    # --------------------------------------------------

    # We create 2 module, to avoid re-compilation. We use 1 set of parameters for both.
    training_model = make_tft_model(config, jit_module=jit_module, dtype=compute_dtype)

    prng_key = jax.random.PRNGKey(config.prng_seed)
    dropout_key, params_key = jax.random.split(prng_key, 2)

    params = training_model.init(params_key, first_x)["params"]

    tx = make_optimizer(config.optimizer, num_training_steps, epochs)

    loss_fn = make_quantile_loss_fn(config.hyperparams.quantiles, dtype=compute_dtype)
    state = TrainStateContainer.create(
        params=params,
        tx=tx,
        apply_fn=training_model.apply,
        dropout_key=dropout_key,
        loss_fn=loss_fn,
        # 1 device is always a GPU
        dtype=compute_dtype,
    )

    hooks = make_training_hooks(
        num_training_steps,
        epochs,
        log_frequency=2,
        logdir=log_dir,
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
    )
    logging.info(f"Finished training with: {training_metrics = }, {validation_metrics = }")
    flax_utils.save_as_msgpack(params, f"{save_path}/{experiment_name}/{tag}/model.msgpack")


def train_on_multiple_devices(
    *,
    data_dir: str,
    batch_size: int,
    experiment_name: Literal["electricity", "favorita"],
    config: ConfigDict,
    epochs: int = 1,
    mixed_precision: bool = False,
    jit_module: bool = False,
    save_path: str = "models",
    tabulate_model: bool = False,
    data_sharding: NamedSharding | None = None,
    params_replication: NamedSharding | None = None,
    device_type: Literal["gpu", "tpu"] = "gpu",
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

    (training_metrics, validation_metrics), params = flax_utils.fit_multi_device(
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
