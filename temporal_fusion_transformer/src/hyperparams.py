from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from absl_extra import flax_utils
import optuna


from temporal_fusion_transformer.src.config_dict import ConfigDict
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.training_lib import make_training_hooks
from temporal_fusion_transformer.src import training_scripts


def optimize_hyperparams(*, config: ConfigDict, data: Tuple[tf.data.Dataset, tf.data.Dataset], epochs: int):
    training_ds, validation_ds = data

    num_steps = int(training_ds.cardinality())

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=1e-3)
        decay_steps = trial.suggest_float("decay_steps", low=0, high=1.0)
        decay_alpha = trial.suggest_float("decay_alpha", low=0.0, high=1.0)
        ema = trial.suggest_float("ema", low=0.0, high=1.0)
        clipnorm = trial.suggest_float("clipnorm", low=0.0, high=1.0)

        latent_size = trial.suggest_int("latent_size", low=5, high=500)
        num_stacks = trial.suggest_int("num_stacks", low=1, high=20)
        num_attention_heads = trial.suggest_int("num_heads", low=1, high=20)

        hyperparams_config = {
            "learning_rate": learning_rate,
            "decay_steps": decay_steps,
            "decay_alpha": decay_alpha,
            "ema": ema,
            "clipnorm": clipnorm,
        }

        def optuna_prune_hooks():
            pass

        hooks = make_training_hooks()

        metrics, _ = training_scripts.train()

        _, validation_metrics = metrics

        return metrics["loss"]

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=num_steps * epochs,
        ),
    )
    study.optimize(objective, n_trials=20)
