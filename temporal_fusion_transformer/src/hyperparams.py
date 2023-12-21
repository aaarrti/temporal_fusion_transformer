# stage 1 -> model size + constant LR, ema, clipnorm
# stage 2 -> weight decay, LR schedule

from __future__ import annotations

import dataclasses
from typing import Literal

# import keras_tuner as kt
import tensorflow as tf
from keras.callbacks import TensorBoard, TerminateOnNaN

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.modeling.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.quantile_loss import QuantileLoss
from temporal_fusion_transformer.src.train_lib import make_optimizer


def optimizer_hyperparameters(
    config: Config,
    dataset: tuple[tf.data.Dataset, tf.data.Dataset],
    logdir: str,
    iteration: Literal[1, 2, 3, "1", "2", "3"] = 1,
    max_epochs: int = 100,
):
    """

    Parameters
    ----------
    config
    dataset
    logdir:
        Directory in which `tensorboard` and `keras_tuner` subdirectories will be created.
    iteration
    max_epochs

    Returns
    -------

    """
    if isinstance(iteration, str):
        iteration = int(iteration)

    if iteration not in (1, 2, 3):
        raise ValueError(f"Unsupported value for `iteration`, supported are: {(1, 2, 3)}")

    training_ds, validation_ds = dataset

    num_train_steps = int(training_ds.cardinality() * config.batch_size)

    def build_model(hp: kt.HyperParameters) -> kt.HyperModel:
        hyper_config = build_fake_config(config, round_1_hparams(hp))
        model = TemporalFusionTransformer.from_dataclass_config(hyper_config)
        model.compile(
            loss=QuantileLoss(config.quantiles),
            jit_compile=False,
            optimizer=make_optimizer(hyper_config, num_train_steps),
        )
        return model

    tuner = kt.Hyperband(
        build_model,
        "val_loss",
        max_epochs=max_epochs,
        project_name="TFT",
        directory=f"{logdir}/keras_tuner",
    )

    tuner.search(
        training_ds,
        validation_data=validation_ds,
        callbacks=[
            TerminateOnNaN(),
            TensorBoard(write_graph=False, log_dir=f"{logdir}/tensorboard"),
        ],
        verbose=0,
    )


def round_1_hparams(hp: kt.HyperParameters) -> dict[str, ...]:
    num_attention_heads = hp.Int("num_attention_heads", 4, 14, step=2)
    num_decoder_blocks = hp.Int("num_decoder_blocks", 1, 12, step=1)
    hidden_layer_size = hp.Int("hidden_layer_size", 8, 256)
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.5)
    learning_rate = hp.Float("learning_rate", 1e-6, 1e-1, sampling="log")

    use_clipnorm = hp.Boolean("use_clipnorm", default=False)

    if hp.conditional_scope(use_clipnorm, True):
        clipnorm = hp.Float("clipnorm", 0, 100, default=0.0)
    else:
        clipnorm = None

    use_ema = hp.Boolean("use_ema", default=False)

    return dict(
        num_attention_heads=num_attention_heads,
        num_decoder_blocks=num_decoder_blocks,
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        use_ema=use_ema,
        clipnorm=clipnorm,
    )


def build_fake_config(config: Config, hparams: dict[str, ...]) -> Config:
    config_dict = dataclasses.asdict(config)
    fake_config = {
        **config_dict,
        **hparams,
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "recurrent_regularizer": None,
    }
    fake_config = Config(**fake_config)
    return fake_config
