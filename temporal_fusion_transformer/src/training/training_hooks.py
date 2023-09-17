from __future__ import annotations

import gc
import os
import platform
from pathlib import Path
from traceback import format_exception
from typing import TypedDict, TYPE_CHECKING
from dataclasses import dataclass

import clu.metric_writers
import clu.periodic_actions
import orbax.checkpoint
import orbax.checkpoint.checkpoint_utils
from absl import logging
from absl_extra.cuda_utils import cuda_devices_available, get_memory_info
from absl_extra.flax_utils import TrainingHooks, combine_hooks, save_as_msgpack
from clu.asynclib import Pool
from clu.metric_writers import SummaryWriter, create_default_writer
from etils import epath
from jaxtyping import Array, Float
from orbax.checkpoint import (
    AsyncCheckpointer,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)

from temporal_fusion_transformer.src.training.training_lib import (
    restore_optimizer_state,
)

if TYPE_CHECKING:
    from absl_extra.flax_utils import StepType
    from temporal_fusion_transformer.src.training.metrics import MetricContainer
    from temporal_fusion_transformer.src.training.training_lib import TrainStateContainer
    from temporal_fusion_transformer.src.modeling.tft_layers import InputStruct

    class LogsDict(TypedDict):
        training_state: TrainStateContainer


__all__ = ["make_training_hooks"]


class CheckpointManager(orbax.checkpoint.CheckpointManager):
    def should_save(self, step: int) -> bool:
        return step % self._options.save_interval_steps == 0 or step in self._options.save_on_steps


pool = clu.asynclib.Pool()


@dataclass(frozen=True, slots=True)
class HooksConfig:
    logdir: str | None
    profile: bool
    checkpoint_directory: str | None
    delete_checkpoints_after_training: bool
    report_progress_frequency: int
    log_metrics_frequency: int
    monitor_exception: bool
    save_path: str
    monitor_gpu_memory: bool

    def make_training_hooks(
        self,
        num_training_steps: int,
        epochs: int,
    ) -> TrainingHooks:
        return make_training_hooks(
            num_training_steps=num_training_steps,
            epochs=epochs,
            checkpoint_directory=self.checkpoint_directory,
            logdir=self.logdir,
            log_metrics_frequency=self.log_metrics_frequency,
            report_progress_frequency=self.report_progress_frequency,
            profile=self.profile,
            monitor_exception=self.monitor_exception,
            monitor_gpu_memory=self.monitor_gpu_memory,
            save_path=self.save_path,
            delete_checkpoints_after_training=self.delete_checkpoints_after_training,
        )


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    profile: bool = False,
    checkpoint_directory: str = "checkpoints",
    delete_checkpoints_after_training: bool = False,
    report_progress_frequency: int = 50,
    log_metrics_frequency: int = 100,
    monitor_exception: bool = True,
    save_path: str | None = None,
    monitor_gpu_memory: bool = True,
) -> TrainingHooks:
    hooks_list = [
        make_metrics_hooks(
            num_training_steps=num_training_steps,
            epochs=epochs,
            logdir=logdir,
            report_progress_frequency=report_progress_frequency,
            log_metrics_frequency=log_metrics_frequency,
        ),
        make_garbage_collection_hooks(),
        make_checkpoint_hooks(
            num_training_steps=num_training_steps,
            epochs=epochs,
            checkpoint_directory=checkpoint_directory,
            delete_checkpoints_after_training=delete_checkpoints_after_training,
            save_path=save_path,
        ),
        make_gpu_memory_monitoring_hook(monitor_gpu_memory),
        make_profiler_hook(profile, logdir),
        make_early_stopping_hook(),
        make_fp_error_hook(monitor_exception),
    ]

    hooks = combine_hooks(*hooks_list)

    def close_pool(*args, **kwargs):
        pool.join()
        pool.close()

    hooks.on_training_end.append(close_pool)

    return hooks


def make_checkpoint_hooks(
    num_training_steps: int,
    epochs: int,
    checkpoint_directory: str | None,
    delete_checkpoints_after_training: bool,
    save_path: str | None,
) -> TrainingHooks:
    hooks = TrainingHooks()
    if checkpoint_directory is None:
        return hooks
    checkpoint_directory = Path(checkpoint_directory).absolute().as_posix()

    options = CheckpointManagerOptions(
        save_interval_steps=200,
        save_on_steps=[num_training_steps * i for i in range(1, epochs)],
        max_to_keep=5,
        cleanup_tmp_directories=True,
        create=True,
        best_mode="min",
        best_fn=lambda metrics: metrics["loss"],
    )
    mngr = CheckpointManager(
        checkpoint_directory,
        AsyncCheckpointer(PyTreeCheckpointHandler(use_ocdbt=True, write_tree_metadata=True)),
        options,
    )

    def checkpoint_fn(step: int, *, training_metrics: MetricContainer, training_state: TrainStateContainer):
        mngr.save(step, training_state, metrics={k: float(v) for k, v in training_metrics.compute().items()})

    def restore_checkpoint(training_state: TrainStateContainer):
        all_steps = mngr.all_steps(True)
        if len(all_steps) == 0:
            return None

        latest_step = max(all_steps)
        restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(training_state)
        restored_dict = mngr.restore(latest_step, restore_kwargs={"restore_args": restore_args})

        restored_optimizer = restore_optimizer_state(training_state.opt_state, restored_dict["opt_state"])
        return training_state.replace(
            rngs=restored_dict["rngs"],
            params=restored_dict["params"],
            step=restored_dict["step"],
            dynamic_scale=restored_dict["dynamic_scale"],
            opt_state=restored_optimizer,
        )

    hooks.on_training_begin.append(restore_checkpoint)
    hooks.on_step_end.append(checkpoint_fn)

    @pool
    def delete_checkpoints(*args, **kwargs):
        for step in mngr.all_steps():
            mngr.delete(step)

    if delete_checkpoints_after_training:
        hooks.on_training_end.append(delete_checkpoints)

    def save_weight_fn(training_state: TrainStateContainer):
        save_as_msgpack(training_state.params, save_path)

    if save_path is not None:
        hooks.on_training_end.append(save_weight_fn)

    return hooks


def make_metrics_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    report_progress_frequency: int = 50,
    log_metrics_frequency: bool = 100,
) -> TrainingHooks:
    logging.info(f"Writing tensorboard logs to {logdir}")

    hooks = TrainingHooks()
    running_on_linux = platform.system().lower() == "linux"

    @pool
    def write_training_metrics_fn(step: int, *args, training_metrics: MetricContainer, **kwargs):
        training_writer.write_scalars(step, training_metrics.compute())

    write_training_metrics = clu.periodic_actions.PeriodicCallback(
        every_steps=10,
        callback_fn=write_training_metrics_fn,
        execute_async=True,
    )

    if running_on_linux:
        training_writer = _create_writer(logdir, "training")
        hooks.on_step_end.append(write_training_metrics)

    training_logger = create_default_writer(None, just_logging=True, collection="training")
    validation_writer = create_default_writer(logdir, just_logging=not running_on_linux, collection="validation")

    @pool
    def log_training_metrics_fn(step: int, *args, training_metrics: MetricContainer, **kwargs):
        training_logger.write_scalars(step, training_metrics.compute())

    @pool
    def write_validation_metrics_fn(epoch: int, *args, validation_metrics: MetricContainer, **kwargs):
        validation_writer.write_scalars(epoch * num_training_steps, validation_metrics.compute())

    log_training_metrics = clu.periodic_actions.PeriodicCallback(
        every_steps=num_training_steps // log_metrics_frequency, callback_fn=log_training_metrics_fn, execute_async=True
    )

    report_progress = clu.periodic_actions.ReportProgress(
        every_steps=num_training_steps // report_progress_frequency,
        num_train_steps=num_training_steps * epochs,
        writer=training_logger,
        every_secs=None,
    )

    @pool
    def report_progress_func(step: int, *args, **kwargs):
        report_progress(step)

    def flush(*args, **kwargs):
        training_writer.flush()
        validation_writer.flush()
        training_logger.flush()

    hooks.on_training_end.append(flush)
    hooks.on_step_end.append(log_training_metrics)
    hooks.on_epoch_end.append(write_validation_metrics_fn)
    hooks.on_step_end.append(report_progress_func)
    return hooks


def make_profiler_hook(profile: bool, logdir: str) -> TrainingHooks:
    hooks = TrainingHooks()

    profiler = clu.periodic_actions.Profile(
        logdir=logdir,
        profile_duration_ms=None,
        every_secs=None,
        first_profile=5,
        every_steps=200,
    )

    @pool
    def call_profiler(step: int, **kwargs):
        profiler(step)

    if profile:
        if platform.system().lower() != "linux":
            logging.warning("Profiling is only supported for linux hosts.")
        else:
            hooks.on_step_begin.append(call_profiler)

    return hooks


def make_garbage_collection_hooks() -> TrainingHooks:
    """We need to manually call python GC to free up XLA memory. See https://github.com/google/jax/issues/14882"""

    @pool
    def gc_func(*args, **kwargs):
        gc.collect()

    periodic_action = clu.periodic_actions.PeriodicCallback(
        every_steps=100, every_secs=5 * 60, execute_async=False, callback_fn=gc_func
    )

    hooks = TrainingHooks()
    hooks.on_step_end.append(periodic_action)
    hooks.on_epoch_end.append(gc_func)
    return hooks


def make_fp_error_hook(monitor_error: bool) -> TrainingHooks:
    def persist_nan_causing_args(
        training_state: TrainStateContainer,
        x_batch: InputStruct,
        y_batch: Float[Array, "batch time n"],
        step_type: StepType,
        exception: Exception,
    ):
        if isinstance(exception, FloatingPointError):
            logging.error(
                f"Step number {int(training_state.step)} failed with on {step_type}_step with {exception} for {x_batch = }, {y_batch = }"
            )
            ex_str = format_exception(exception)
            data = {
                "state": training_state,
                "x_batch": x_batch,
                "y_batch": y_batch,
                "exception": ex_str,
                "step_type": step_type,
            }
            save_as_msgpack(data, "fp_error_data.msgpack")
        raise

    hooks = TrainingHooks()
    if monitor_error:
        hooks.on_error.append(persist_nan_causing_args)
    return hooks


def _create_writer(logdir: str, collection: str) -> SummaryWriter:
    logdir = epath.Path(logdir)
    logdir /= collection
    return SummaryWriter(os.fspath(logdir))


def make_early_stopping_hook() -> TrainingHooks:
    def replace_early_stopping(
        *args, training_state: TrainStateContainer, training_metrics: MetricContainer, **kwargs
    ) -> LogsDict | None:
        if training_state.early_stopping is not None:
            early_stopping = training_state.early_stopping.update(training_metrics.compute()["loss"])[1]
            return {"training_state": training_state.replace(early_stopping=early_stopping)}

    hooks = TrainingHooks()
    hooks.on_step_end.append(replace_early_stopping)

    return hooks


def make_gpu_memory_monitoring_hook(monitor_gpu_memory: bool) -> TrainingHooks:
    hooks = TrainingHooks()

    @pool
    def monitor_fn(*args, step: int, **kwargs):
        memory = get_memory_info()
        usage_details = [f"{float(i.used):.2f}/{i.total} GB" for i in memory]
        logging.info(f"{step = }, memory usage = {usage_details}")

    callback = clu.periodic_actions.PeriodicCallback(
        every_steps=None, every_secs=10 * 60, on_steps=[0, 5], callback_fn=monitor_fn, pass_step_and_time=True
    )

    if monitor_gpu_memory and cuda_devices_available():
        hooks.on_step_begin.append(callback)

    return hooks
