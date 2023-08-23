from __future__ import annotations

import functools
from typing import Literal, Mapping, Protocol, Tuple

import clu.periodic_actions
import jax
import optax
import tensorflow as tf
from absl import logging
from absl_extra import flax_utils
from absl_extra.typing_utils import ParamSpec
from clu.metrics import Average
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass, field, PyTreeNode
from flax.training.dynamic_scale import DynamicScale
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, PRNGKeyArray, Scalar, jaxtyped
from orbax.checkpoint import Checkpointer, CheckpointManagerOptions, CheckpointManager, PyTreeCheckpointHandler

from temporal_fusion_transformer.src.config_dict import OptimizerConfig
from temporal_fusion_transformer.src.quantile_loss import QuantileLossFn
from temporal_fusion_transformer.src.tft_layers import InputStruct

P = ParamSpec("P")


class EarlyStoppingWrapper(PyTreeNode):
    
    early_stopping: EarlyStopping
    
    @property
    def should_stop(self):
        return self.early_stopping.should_stop
    
    def __call__(self, *args, training_metrics: MetricContainer, **kwargs):
        self.early_stopping.update(training_metrics.compute()['loss'])


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
    dynamic_scale: DynamicScale | None = None


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    log_frequency: int | Tuple[int, int] = 10,
    add_early_stopping: bool = True,
    profile: bool = False,
    checkpoint_frequency: int = 3,
    checkpoint_directory: str = "checkpoints",
    delete_checkpoints_after_training: bool = True,
) -> flax_utils.TrainingHooks:
    logging.info(f"Writing tensorboard logs to {logdir}")
    
    if isinstance(log_frequency, Tuple):
        write_metrics_frequency, report_progress_frequency = log_frequency
    else:
        write_metrics_frequency, report_progress_frequency = log_frequency, log_frequency

    hooks = flax_utils.make_training_hooks(
        num_training_steps=num_training_steps,
        epochs=epochs,
        write_metrics_frequency=write_metrics_frequency,
        report_progress_frequency=report_progress_frequency,
        tensorboard_logdir=logdir,
    )
    if add_early_stopping:
        early_stopping = EarlyStopping(patience=100, min_delta=0.1)
        early_stopping = EarlyStoppingWrapper(early_stopping=early_stopping)
        hooks.on_step_end.append(early_stopping)

    if profile:
        profiler = clu.periodic_actions.Profile(logdir=logdir, profile_duration_ms=None)

        def call_profiler(step: int, **kwargs):
            profiler(step)

        hooks.on_step_begin.append(call_profiler)

    add_checkpoint = checkpoint_directory is not None

    if add_checkpoint:
        options = CheckpointManagerOptions(
            save_interval_steps=checkpoint_frequency,
            max_to_keep=5,
            cleanup_tmp_directories=True,
            best_mode="min",
            best_fn=lambda metrics: metrics["loss"],
        )
        mngr = CheckpointManager(
            checkpoint_directory,
            Checkpointer(PyTreeCheckpointHandler(use_ocdbt=True)),
            options,
        )

        def checkpoint_fn(step: int, *, training_metrics: MetricContainer, training_state: TrainStateContainer):
            mngr.save(step, training_state, metrics=training_metrics.compute())

        def force_checkpoint_fn(step: int, *, training_metrics: MetricContainer, training_state: TrainStateContainer):
            mngr.save(step, training_state, metrics=training_metrics.compute(), force=True)

        def restore_checkpoint(*args, training_state: TrainStateContainer, **kwargs) -> TrainStateContainer | None:
            all_steps = mngr.all_steps(True)
            if len(all_steps) == 0:
                return None

            latest_step = max(all_steps)
            state = mngr.restore(latest_step, items=training_state)
            return state

        hooks.on_training_begin.append(restore_checkpoint)

        hooks.on_step_end.append(checkpoint_fn)
        hooks.on_epoch_end.append(force_checkpoint_fn)

        if delete_checkpoints_after_training:

            def delete_checkpoints(*args, **kwargs):
                for step in mngr.all_steps():
                    mngr.delete(step)

            hooks.on_training_end.append(delete_checkpoints)

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
        state = state.apply_gradients(grads=grads)
    else:
        dynamic_scale, is_fin = None, None
        loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    if state.dynamic_scale is not None:
        select_fn = Partial(jnp.where, is_fin)
        state = state.replace(
            opt_state=jax.tree_util.tree_map(select_fn, state.opt_state, state.opt_state),
            params=jax.tree_util.tree_map(select_fn, state.params, state.params),
        )

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
        y = state.apply_fn({"params": params}, x_batch, True, rngs={"dropout": dropout_train_key})
        y_loss = state.loss_fn(y_batch, y).mean()
        return y_loss

    if state.dynamic_scale is not None:
        dynamic_scale, is_fin, loss, grads = state.dynamic_scale.value_and_grad(loss_fn, axis_name="i")(state.params)
        state.replace(dynamic_scale=dynamic_scale)
    else:
        dynamic_scale, is_fin = None, None
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name="i")

    state = state.apply_gradients(grads=grads)
    if state.dynamic_scale is not None:
        select_fn = Partial(jnp.where, is_fin)
        state = state.replace(
            opt_state=jax.tree_util.tree_map(select_fn, state.opt_state, state.opt_state),
            params=jax.tree_util.tree_map(select_fn, state.params, state.params),
        )

    metrics = MetricContainer.gather_from_model_output(loss=loss, axis_name="i")
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
    data_dir: str, batch_size: int, prng_seed: int, shuffle_buffer_size: int = 2048, dtype=jnp.float32
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """

    Parameters
    ----------
    data_dir
    batch_size
    shuffle_buffer_size:
        If set to None, will do a full-reshuffle.
    prng_seed
    dtype

    Returns
    -------

    """

    tf_dtype = tf.dtypes.as_dtype(dtype)

    def downcast_input(x, y):
        return tf.cast(x, tf_dtype), tf.cast(y, tf_dtype)

    def load_fn(split: Literal["training", "validation"], buffer_size: int) -> tf.data.Dataset:
        ds = tf.data.Dataset.load(f"{data_dir}/{split}", compression="GZIP")

        if buffer_size == -1:
            buffer_size = int(ds.cardinality())

        return (
            ds.shuffle(buffer_size, seed=prng_seed, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
            .map(downcast_input)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    training_ds = load_fn("training", shuffle_buffer_size)
    validation_ds = load_fn("validation", shuffle_buffer_size)
    return training_ds, validation_ds


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
