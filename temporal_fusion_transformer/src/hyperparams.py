from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import jax
import jax.numpy as jnp
import optuna
from absl_extra import flax_utils
from ml_collections import ConfigDict

from temporal_fusion_transformer.src import training_scripts
from temporal_fusion_transformer.src.config_dict import (
    ConfigDictProto,
    DatasetConfig,
)
from temporal_fusion_transformer.src.training_lib import MetricContainer, load_dataset

if TYPE_CHECKING:
    import tensorflow as tf


def optimize_experiment_hyperparams(
    experiment_name: str,
    config: ConfigDictProto | ConfigDict,
    epochs: int,
    batch_size: int,
    data_dir: str,
    device_type: Literal["gpu", "tpu"],
    jit_module: bool = False,
    mixed_precision: bool = True,
    n_trials: int | None = 20,
    n_jobs: int = 1,
    verbose: bool = False,
):
    if mixed_precision:
        if device_type == "gpu":
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.bfloat16
    else:
        compute_dtype = jnp.float32

    num_devices = jax.device_count()

    data = load_dataset(
        f"{data_dir}/{experiment_name}",
        batch_size * num_devices,
        config.prng_seed,
        dtype=compute_dtype,
        shuffle_buffer_size=config.shuffle_buffer_size,
        num_encoder_steps=config.fixed_params.num_encoder_steps,
    )

    optimize_hyperparams(
        fixed_params=config.fixed_params,
        data=data,
        epochs=epochs,
        experiment_name=experiment_name,
        batch_size=batch_size,
        device_type=device_type,
        jit_module=jit_module,
        mixed_precision=mixed_precision,
        n_trials=n_trials,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def optimize_hyperparams(
    *,
    fixed_params: ConfigDict | DatasetConfig,
    data: Tuple[tf.data.Dataset, tf.data.Dataset],
    epochs: int,
    batch_size: int,
    device_type: Literal["gpu", "tpu"],
    jit_module: bool = False,
    mixed_precision: bool = True,
    n_trials: int | None = 20,
    n_jobs: int = 1,
    verbose: bool = False,
    experiment_name: str,
):
    training_ds, validation_ds = data
    num_training_steps = int(training_ds.cardinality())

    fixed_params = fixed_params.to_dict()

    def objective(trial: optuna.Trial) -> float:
        def hyperband_pruning_hook(step: int, *, training_metrics: MetricContainer, **kwargs):
            loss = training_metrics.compute()["loss"]
            trial.report(loss, int(step))

            if trial.should_prune():
                raise optuna.TrialPruned()

        hooks = flax_utils.make_training_hooks(
            num_training_steps=num_training_steps,
            epochs=epochs,
            report_progress_frequency=10,
            write_metrics_frequency=None,
            tensorboard_logdir=None,
        )
        hooks.on_step_end.append(hyperband_pruning_hook)
        hooks.on_error.append(prune_trial_on_nan)

        learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=1e-3)
        decay_steps = trial.suggest_float("decay_steps", low=0, high=1.0)
        decay_alpha = trial.suggest_float("decay_alpha", low=0.0, high=1.0)
        ema = trial.suggest_float("ema", low=0.0, high=1.0)
        clipnorm = trial.suggest_float("clipnorm", low=0.0, high=1.0)

        latent_dim = trial.suggest_int("latent_dim", low=5, high=500)
        num_stacks = trial.suggest_int("num_stacks", low=1, high=20)
        num_attention_heads = trial.suggest_int("num_heads", low=1, high=20)

        dropout_rate = trial.suggest_float("dropout_rate", low=0.1, high=0.3, step=0.05)
        attention_dropout_rate = trial.suggest_float("attention_dropout_rate", low=0.1, high=0.3, step=0.05)

        if latent_dim % num_attention_heads != 0:
            raise optuna.TrialPruned("`latent_dim` must be divisible by `num_attention_heads`")

        pseudo_config = ConfigDict(
            {
                "fixed_params": fixed_params,
                "optimizer": {
                    "learning_rate": learning_rate,
                    "decay_steps": decay_steps,
                    "decay_alpha": decay_alpha,
                    "ema": ema,
                    "clipnorm": clipnorm,
                },
                "model": {
                    "quantiles": [0.1, 0.5, 0.9],
                    "dropout_rate": dropout_rate,
                    "latent_dim": latent_dim,
                    "num_attention_heads": num_attention_heads,
                    "num_decoder_blocks": num_stacks,
                    "attention_dropout_rate": attention_dropout_rate,
                },
            }
        )

        metrics, _ = training_scripts.train(
            data=data,
            config=pseudo_config,
            batch_size=batch_size,
            verbose=False,
            device_type=device_type,
            hooks=hooks,
            jit_module=jit_module,
            mixed_precision=mixed_precision,
            early_stopping=None,
        )
        # Ideally, we should use validation metrics, but this would be way too time-consuming.
        return metrics[0]["loss"]

    study = optuna.create_study(
        direction="minimize",
        load_if_exists=True,
        storage="sqlite:///optuna.db",
        study_name=experiment_name,
        pruner=optuna.pruners.HyperbandPruner(max_resource=num_training_steps * epochs),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )


def prune_trial_on_nan(*args, exception: Exception):
    if isinstance(exception, FloatingPointError):
        raise optuna.TrialPruned("Encountered NaN")
    else:
        raise
