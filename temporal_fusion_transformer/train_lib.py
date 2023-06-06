from __future__ import annotations

from typing import Callable, Sequence, Dict, Tuple
import numpy as np

import keras_tuner as kt
import tensorflow as tf
from keras.callbacks import TensorBoard, TerminateOnNaN, BackupAndRestore
from keras.losses import Loss
from keras.utils.tf_utils import can_jit_compile
from keras import mixed_precision

from temporal_fusion_transformer.modeling import TFTInputs


def train_with_fixed_hyper_parameters(
    model_factory: Callable[[], tf.keras.Model],
    optimizer_factory: Callable[[], tf.keras.optimizers.Optimizer],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 1,
) -> Tuple[tf.keras.Model, Dict[str, np.ndarray]]:
    if can_jit_compile():
        tf.config.optimizer.set_jit("autoclustering")
        mixed_precision.set_global_policy("mixed_float16")

    model = model_factory()
    model.compile(
        optimizer=optimizer_factory(),
        loss=QuantileLoss(model.quantiles),
        jit_compile=can_jit_compile(True),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            TensorBoard("tensorboard_logs", write_graph=False),
            TerminateOnNaN(),
            BackupAndRestore("checkpoints"),
        ],
    ).history
    return model, history


def fine_tune_hyper_parameters(
    model_factory: Callable[[kt.HyperParameters], tf.keras.Model],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 1,
    max_trials: int = 20,
    prng_seed: int = 42,
    name: str = "experiment",
) -> kt.HyperParameters:
    if can_jit_compile():
        tf.config.optimizer.set_jit("autoclustering")
        mixed_precision.set_global_policy("mixed_float16")

    tuner = kt.RandomSearch(
        hypermodel=model_factory,
        objective="val_loss",
        seed=prng_seed,
        max_trials=max_trials,
        overwrite=True,
        directory="tuner_logs",
        project_name=name,
    )
    tuner.search_space_summary()
    tuner.search(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            TensorBoard("tensorboard_logs", write_graph=False),
            TerminateOnNaN(),
        ],
        verbose=2,
    )
    return tuner.get_best_hyperparameters(0)[0]


class QuantileLoss(Loss):
    """
    Computes the combined quantile loss for specified quantiles.

    References:
        - https://arxiv.org/abs/1912.09363

    """

    def __init__(self, quantiles: Sequence[int], output_size: int = 1):
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
        self.quantiles = tf.constant(quantiles)
        self.output_size = tf.constant(output_size)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """

        Parameters
        ----------
        y_true:
            Targets of shape (batch, time_steps, 1).
        y_pred:
            Predictions of shape (batch, time_steps, quantiles).

        Returns
        -------

        """

        if tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)

        # @tf.function(
        #    reduce_retracing=True,
        #    jit_compile=can_jit_compile(True),
        #    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
        # )
        def inner_fn(q: int):
            return quantile_loss(
                y_true,
                y_pred[..., self.output_size * q : self.output_size * (q + 1)],
                self.quantiles[q],
            )

        loss = tf.vectorized_map(inner_fn, tf.range(len(self.quantiles)))
        return tf.reduce_sum(loss, axis=0)


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


def load_data_from_archive(path: str) -> Dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    data = {}

    for k in (
        "identifier",
        "time",
        "outputs",
        "inputs_static",
        "inputs_known_real",
        "inputs_known_categorical",
        "inputs_observed",
    ):
        if k in archive:
            data[k] = archive[k]

    return data


def make_input_tuple(data: Dict[str, tf.Tensor]) -> Tuple[TFTInputs, tf.Tensor]:
    return (
        TFTInputs(
            static=data["inputs_static"],
            known_real=data["inputs_known_real"],
            known_categorical=data.get("inputs_known_categorical"),
            observed=data.get("inputs_observed"),
        ),
        data["outputs"],
    )
