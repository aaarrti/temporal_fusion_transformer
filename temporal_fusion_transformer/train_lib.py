from __future__ import annotations

from typing import Callable, Sequence, Dict, Tuple, TYPE_CHECKING, Mapping, Optional

from glob import glob
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, TerminateOnNaN, BackupAndRestore
from keras.losses import Loss
from keras.utils.tf_utils import can_jit_compile
import keras_tuner as kt
from tensorflow.python.framework.dtypes import DType

if TYPE_CHECKING:
    from temporal_fusion_transformer.modeling import (
        TemporalFusionTransformer,
        TFTInputs,
    )


def train_with_fixed_hyper_parameters(
    model_factory: Callable[[], TemporalFusionTransformer],
    optimizer_factory: Callable[[], tf.keras.optimizers.Optimizer],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    **kwargs,
) -> Tuple[TemporalFusionTransformer, Dict[str, np.ndarray]]:
    if can_jit_compile():
        tf.config.optimizer.set_jit("autoclustering")

    model = model_factory()
    model.compile(
        optimizer=optimizer_factory(),
        loss=QuantileLoss(model.quantiles),
        jit_compile=can_jit_compile(True),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[
            TensorBoard("tensorboard_logs", write_graph=True),
            TerminateOnNaN(),
            BackupAndRestore("checkpoints"),
        ],
        **kwargs,
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


def load_sharded_dataset(
    path: str,
    batch_size: int,
    element_spec: Mapping[str, tf.TensorSpec] | Tuple[tf.TensorSpec, ...] | None = None,
    map_fn: Optional[
        Callable[[Mapping[str, tf.Tensor]], Tuple[TFTInputs, tf.Tensor]]
    ] = None,
    dtype: DType = tf.float32,
    cache_filename: str = "",
) -> tf.data.Dataset:
    """

    Parameters
    ----------
    path:
        Path to local filesystem location, where sharded datasets is saved.
    batch_size:
    element_spec:
        While loading shards, it is required to provided element spec, default
        ```
        {
            "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
            "time": tf.TensorSpec([None, 192, 1], dtype=tf.float64),
            "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float64),
            "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int64),
            "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float64),
            "inputs_known_categorical": tf.TensorSpec([None, 192, 3], dtype=tf.int64),
            "inputs_observed": tf.TensorSpec([None, 192, 3], dtype=tf.float64),
        }
        ```
    map_fn:
        Function used to map raw dataset entries to (x, y) input tuples used for training.
    dtype:
        Floating point data type, default=tf.float32.
    cache_filename:
        It is likely, that dataset, won't fit into memory, provide this argument to cache it in local file system.

    Returns
    -------

    retval:
        tf.data.Dataset ready to use for training.

    """
    # Import actual type during runtime
    from temporal_fusion_transformer.modeling import TFTInputs

    if element_spec is None:
        element_spec = {
            "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
            "time": tf.TensorSpec([None, 192, 1], dtype=tf.float64),
            "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float64),
            "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int64),
            "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float64),
            "inputs_known_categorical": tf.TensorSpec([None, 192, 3], dtype=tf.int64),
            "inputs_observed": tf.TensorSpec([None, 192, 3], dtype=tf.float64),
        }

    def default_map_fn(arg):
        return (
            TFTInputs(
                static=tf.cast(arg["inputs_static"], tf.int32),
                known_real=tf.cast(arg["inputs_known_real"], dtype),
                known_categorical=tf.cast(arg["inputs_known_categorical"], tf.int32),
                observed=tf.cast(arg["inputs_observed"], dtype),
            ),
            tf.cast(arg["outputs"], dtype),
        )

    if map_fn is None:
        map_fn = default_map_fn

    return (
        tf.data.Dataset.from_tensor_slices(glob(f"{path}/*"))
        .flat_map(
            lambda i: tf.data.Dataset.load(i, element_spec=element_spec).map(map_fn),
        )
        .rebatch(batch_size)
        .cache(cache_filename)
        .prefetch(tf.data.AUTOTUNE)
    )
