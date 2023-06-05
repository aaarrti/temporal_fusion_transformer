from __future__ import annotations

from typing import List, NamedTuple, Type

import tensorflow as tf
from keras.callbacks import TensorBoard, TerminateOnNaN, BackupAndRestore
from keras.losses import Loss
from keras.utils.tf_utils import can_jit_compile

from temporal_fusion_transformer.experiments import Experiment
from temporal_fusion_transformer.modeling import TemporalFusionTransformer

"""
- x_batch -> TFTInputs
- y_batch -> (batch_size, time_steps - n_encoder_steps, n_outputs)
    we must concatenate 3 of those, to use for 3 different quantiled losses
- sample_weights -> (batch, n_outputs) (all ones ??)
"""


class HyperParameters(NamedTuple):
    max_gradient_norm: float
    learning_rate: float


def train_with_default_hyper_parameters(
    experiment: Type[Experiment],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    hp: HyperParameters,
    epochs: int = 1,
) -> tf.keras.Model:
    model = TemporalFusionTransformer(
        static_categories_sizes=experiment.data_params.static_categories_sizes,
        known_categories_sizes=experiment.data_params.known_categories_sizes,
        num_encoder_steps=experiment.data_params.num_encoder_steps,
        num_attention_heads=experiment.model_params.num_attention_heads,
        dropout_rate=experiment.model_params.dropout_rate,
        hidden_layer_size=experiment.model_params.hidden_layer_size,
        output_size=experiment.data_params.num_outputs,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            # TODO: cosine schedule?
            learning_rate=hp.learning_rate,
            clipnorm=hp.max_gradient_norm,
        ),
        loss=QuantileLoss(model.quantiles),
        jit_compile=can_jit_compile(True),
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            TensorBoard("tensorboard_logs", write_graph=False),
            TerminateOnNaN(),
            BackupAndRestore("checkpoints"),
        ],
    )
    return model


# def fine_tune_hyper_parameters(
#    experiment: Experiment,
#    batch_size: int = 32,
#    epochs: int = 1,
#    prng_seed: int = 42,
#    max_trials: int = 20,
# ):
#    import keras_tuner
#
#    train_ds, val_ds = experiment.train_test_split
#    train_ds = make_dataset(train_ds, batch_size)
#    val_ds = make_dataset(val_ds, batch_size)
#
#    def make_model(hp: keras_tuner.HyperParameters) -> keras_tuner.HyperModel:
#        model = TemporalFusionTransformer(
#            static_categories_sizes=experiment.fixed_params.static_categories_sizes,
#            known_categories_sizes=experiment.fixed_params.known_categories_sizes,
#            num_encoder_steps=experiment.default_hyperparams.num_encoder_steps,
#            num_attention_heads=experiment.default_hyperparams.num_attention_heads,
#            dropout_rate=experiment.default_hyperparams.dropout_rate,
#            hidden_layer_size=experiment.default_hyperparams.hidden_layer_size,
#            output_size=experiment.fixed_params.num_outputs,
#            quantiles=experiment.default_hyperparams.quantiles,
#        )
#
#        learning_rate = hp.Float(
#            "learning_rate", max_value=1e-2, min_value=1e-3, sampling="log", step=0.2
#        )
#        max_gradient_norm = hp.Float(
#            "max_grad_norm", max_value=0.1, min_value=1e-3, sampling="log", step=0.2
#        )
#
#        model.compile(
#            optimizer=tf.keras.optimizers.Adam(
#                # TODO: cosine schedule?
#                learning_rate=learning_rate,
#                clipnorm=max_gradient_norm,
#            ),
#            loss=QuantileLoss(experiment.default_hyperparams.quantiles),
#            jit_compile=can_jit_compile(True),
#        )
#
#    tuner = keras_tuner.RandomSearch(
#        hypermodel=make_model,
#        objective="val_loss",
#        seed=prng_seed,
#        max_trials=max_trials,
#        overwrite=True,
#        directory="tuner_logs",
#        project_name=experiment.name,
#    )
#    tuner.search_space_summary()
#    tuner.search(
#        train_ds,
#        epochs=epochs,
#        validation_data=val_ds,
#        callbacks=[
#            TensorBoard("tensorboard_logs", write_graph=False),
#            TerminateOnNaN(),
#        ],
#        verbose=2,
#    )
#
#


class QuantileLoss(Loss):
    """
    Computes the combined quantile loss for specified quantiles.

    References:
        - https://arxiv.org/abs/1912.09363

    """

    def __init__(self, quantiles: List[int], output_size: int = 1):
        """

        Parameters
        ----------
        quantiles:
            Quantiles to compute losses
        """
        super().__init__()
        for quantile in quantiles:
            if quantile < 0 or quantile > 1:
                raise ValueError(
                    f"Illegal quantile value={quantile}! Values should be between 0 and 1."
                )
        self.quantiles = quantiles
        self.output_size = output_size

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """

        Parameters
        ----------
        y_true:
            Targets
        y_pred:
            Predictions

        Returns
        -------

        """
        quantiles_used = set(self.quantiles)

        loss = 0.0
        for i, quantile in enumerate(self.quantiles):
            if quantile in quantiles_used:
                loss += quantile_loss(
                    y_true[..., self.output_size * i : self.output_size * (i + 1)],
                    y_pred[Ellipsis, self.output_size * i : self.output_size * (i + 1)],
                    tf.constant(quantile),
                )
        return loss


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile(True))
def quantile_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, quantile: tf.Tensor
) -> tf.Tensor:
    """
    Parameters
    ----------
    y_true:
        Targets
    y_pred:
        Predictions
    quantile:
        Quantile to use for loss calculations (between 0 & 1)

    Returns
    -------
    loss:
        Loss value.
    """
    prediction_underflow = y_true - y_pred
    positive_prediction_underflow = tf.maximum(prediction_underflow, 0.0)
    negative_prediction_underflow = tf.maximum(-prediction_underflow, 0.0)
    q_loss = (
        quantile * positive_prediction_underflow
        + (1.0 - quantile) * negative_prediction_underflow
    )
    return tf.reduce_sum(q_loss, axis=-1)
