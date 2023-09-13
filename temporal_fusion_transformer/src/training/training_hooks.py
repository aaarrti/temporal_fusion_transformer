from __future__ import annotations

import os
import platform
from pathlib import Path
from traceback import format_exception

import clu.metric_writers
import clu.periodic_actions
import orbax.checkpoint
import orbax.checkpoint.checkpoint_utils
from absl import logging
from absl_extra import flax_utils
from clu.metric_writers import AsyncMultiWriter, SummaryWriter, create_default_writer
from etils import epath
from jaxtyping import Array, Float
from orbax.checkpoint import (
    AsyncCheckpointer,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)

from temporal_fusion_transformer.src.modeling.tft_layers import InputStruct
from temporal_fusion_transformer.src.training.metrics import MetricContainer
from temporal_fusion_transformer.src.training.training_lib import (
    TrainStateContainer,
    restore_optimizer_state,
)


class CheckpointManager(orbax.checkpoint.CheckpointManager):
    def should_save(self, step: int) -> bool:
        return step % self._options.save_interval_steps == 0 or step in self._options.save_on_steps


def make_training_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    profile: bool = False,
    checkpoint_directory: str = "checkpoints",
    delete_checkpoints_after_training: bool = False,
    report_progress_frequency: int = 50,
    log_metrics_frequency: bool = 100,
    monitor_exception: bool = True,
    save_path: str | None = None,
) -> flax_utils.TrainingHooks:
    hooks_list = [
        make_metrics_hooks(num_training_steps, epochs, logdir, report_progress_frequency, log_metrics_frequency),
    ]

    if checkpoint_directory is not None:
        hooks_list.append(
            make_checkpoint_hooks(
                num_training_steps, epochs, checkpoint_directory, delete_checkpoints_after_training, save_path
            )
        )

    if profile:
        hooks_list.append(make_profiler_hooks(logdir))

    if len(hooks_list) > 1:
        hooks = flax_utils.combine_hooks(*hooks_list)
    else:
        hooks = hooks_list[0]

    if monitor_exception:
        hooks.on_error.append(persist_nan_causing_args)

    hooks.on_step_end.append(maybe_replace_early_stopping)
    return hooks


def make_checkpoint_hooks(
    num_training_steps: int,
    epochs: int,
    checkpoint_directory: str,
    delete_checkpoints_after_training: bool,
    save_path: str | None,
) -> flax_utils.TrainingHooks:
    hooks = flax_utils.TrainingHooks()
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
            dropout_key=restored_dict["dropout_key"],
            params=restored_dict["params"],
            step=restored_dict["step"],
            dynamic_scale=restored_dict["dynamic_scale"],
            opt_state=restored_optimizer,
        )

    hooks.on_training_begin.append(restore_checkpoint)
    hooks.on_step_end.append(checkpoint_fn)

    if delete_checkpoints_after_training:

        def delete_checkpoints(*args, **kwargs):
            for step in mngr.all_steps():
                mngr.delete(step)

        hooks.on_training_end.append(delete_checkpoints)

    if save_path is not None:

        def save_weight_fn(training_state: TrainStateContainer):
            flax_utils.save_as_msgpack(training_state.params, save_path)

        hooks.on_training_end.append(save_weight_fn)

    return hooks


def make_metrics_hooks(
    num_training_steps: int,
    epochs: int,
    logdir: str,
    report_progress_frequency: int = 50,
    log_metrics_frequency: bool = 100,
) -> flax_utils.TrainingHooks:
    logging.info(f"Writing tensorboard logs to {logdir}")

    hooks = flax_utils.TrainingHooks()
    running_on_linux = platform.system().lower() == "linux"

    if running_on_linux:
        training_writer = create_writer(logdir, "training")

        def write_training_metrics_fn(step: int, *args, training_metrics: MetricContainer, **kwargs):
            training_writer.write_scalars(step, training_metrics.compute())

        write_training_metrics = clu.periodic_actions.PeriodicCallback(
            every_steps=10,
            callback_fn=write_training_metrics_fn,
            execute_async=True,
        )

        hooks.on_step_end.append(write_training_metrics)

    training_logger = create_default_writer(None, just_logging=True, collection="training")
    validation_writer = create_default_writer(logdir, just_logging=not running_on_linux, collection="validation")

    def log_training_metrics_fn(step: int, *args, training_metrics: MetricContainer, **kwargs):
        training_logger.write_scalars(step, training_metrics.compute())

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


def make_profiler_hooks(logdir: str) -> flax_utils.TrainingHooks:
    hooks = flax_utils.TrainingHooks()
    if platform.system().lower() != "linux":
        logging.warning("Profiling is only supported for linux hosts.")
    else:
        profiler = clu.periodic_actions.Profile(
            logdir=logdir,
            profile_duration_ms=None,
            every_secs=None,
            first_profile=5,
            every_steps=200,
        )

        def call_profiler(step: int, **kwargs):
            profiler(step)

        hooks.on_step_begin.append(call_profiler)

    return hooks


def persist_nan_causing_args(
    training_state: TrainStateContainer,
    x_batch: InputStruct,
    y_batch: Float[Array, "batch time n"],
    step_type: flax_utils.StepType,
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
        flax_utils.save_as_msgpack(data, "fp_error_data.msgpack")
    raise


def create_writer(logdir: str, collection: str) -> AsyncMultiWriter:
    logdir = epath.Path(logdir)
    logdir /= collection
    writers = [SummaryWriter(os.fspath(logdir))]
    return AsyncMultiWriter(writers)


def maybe_replace_early_stopping(
    *args, training_state: TrainStateContainer, training_metrics: MetricContainer, **kwargs
):
    if training_state.early_stopping is not None:
        early_stopping = training_state.early_stopping.update(training_metrics.compute()["loss"])[1]
        return {"training_state": training_state.replace(early_stopping=early_stopping)}