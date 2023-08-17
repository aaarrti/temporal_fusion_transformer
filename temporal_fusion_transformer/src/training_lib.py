from __future__ import annotations

import functools
import platform
from typing import Callable, Generator, Literal, Mapping, Protocol, Tuple

import clu.periodic_actions
import jax
import optax
import tensorflow as tf
from absl import logging
from absl_extra import flax_utils
from absl_extra.typing_utils import ParamSpec
from clu.metrics import Average
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass, field
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, PRNGKeyArray, Scalar, jaxtyped

from temporal_fusion_transformer.src.config_dict import FixedParamsConfig, OptimizerConfig
from temporal_fusion_transformer.src.quantile_loss import QuantileLossFn
from temporal_fusion_transformer.src.tft_layers import ComputeDtype, InputStruct
from temporal_fusion_transformer.src.utils import make_input_struct_from_config

P = ParamSpec("P")


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
    # TODO: save learning_rate & loss_scale


@jaxtyped
class TrainStateContainer(TrainState):
    apply_fn: ApplyFunc = field(pytree_node=False)
    loss_fn: QuantileLossFn = field(pytree_node=False)
    dropout_key: PRNGKeyArray
    dynamic_scale: DynamicScale | None


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    log_frequency: int = 10,
    add_early_stopping: bool = True,
    add_checkpoint: bool = False,
    profile: bool = False,
    checkpoint_frequency: int = 3,
) -> flax_utils.TrainingHooks:
    hooks = flax_utils.make_training_hooks(
        num_training_steps=num_training_steps,
        epochs=epochs,
        write_metrics_frequency=log_frequency,
        report_progress_frequency=log_frequency,
        tensorboard_logdir=logdir,
    )
    if add_early_stopping:
        early_stopping = EarlyStopping(patience=int(num_training_steps // 4), min_delta=0.1)

        def update_early_stopping(*args, training_metrics: MetricContainer, **kwargs):
            loss = training_metrics.compute()["loss"]
            early_stopping.update(loss)

        def reset_early_stopping(*args, **kwargs):
            early_stopping.reset()

        hooks.on_step_end.append(update_early_stopping)
        hooks.on_epoch_end.append(reset_early_stopping)

    if profile:
        if platform.system().lower() == "linux":
            profiler = clu.periodic_actions.Profile(logdir=logdir, profile_duration_ms=None)

            def call_profiler(step: int, **kwargs):
                profiler(step)

            hooks.on_step_begin.append(call_profiler)
        else:
            logging.warning("Profiling is only supported for linux hosts.")

    if add_checkpoint:
        # TODO: add checkpoint
        pass

    return hooks


@jaxtyped
@jax.jit
def single_device_train_step(
    state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
) -> Tuple[TrainStateContainer, MetricContainer]:
    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

    def loss_fn(params: FrozenDict) -> Float[Scalar]:
        # pass training=True as positional args, since flax.nn.jit does not support kwargs.
        y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.dynamic_scale is not None:
        # loss scaling logic is taken from https://github.com/google/flax/blob/main/examples/wmt/train.py#L177
        dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(loss_fn)(state.params)
        state.replace(dynamic_scale=dynamic_scale)
        new_state = state.apply_gradients(grads=grads)
        select_fn = Partial(jnp.where, is_fin)
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(select_fn, new_state.opt_state, state.opt_state),
            params=jax.tree_util.tree_map(select_fn, new_state.params, state.params),
        )
    else:
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)

    metrics = MetricContainer.single_from_model_output(loss=loss)
    return new_state, metrics


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
        y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.dynamic_scale is not None:
        _, _, loss, grads = state.dynamic_scale.value_and_grad(loss_fn, axis_name="i")(state.params)
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
    loss = lax.pmean(loss, axis_name="i")
    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
    return metrics


def load_dataset(
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    prng_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def load_fn(split: Literal["training", "validation"]) -> tf.data.Dataset:
        return (
            tf.data.Dataset.load(f"{data_dir}/{split}", compression="GZIP")
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer_size, seed=prng_seed)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training")
    validation_ds = load_fn("validation")

    if platform.system().lower() == "darwin":
        logging.warning("Running on MacOS, will use 10 training and 2 validation batches for testing purposes.")
        training_ds = training_ds.take(10)
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
