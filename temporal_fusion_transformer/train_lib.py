from __future__ import annotations

import functools
from glob import glob
from typing import Callable, Sequence, Dict, Tuple, Mapping, Optional, Literal
from importlib import util


import numpy as np
import tensorflow as tf
from absl_extra.collection_utils import map_dict
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.losses import Loss
from keras.utils.tf_utils import can_jit_compile
from sklearn.utils import gen_batches
from tensorflow.python.framework.dtypes import DType


if util.find_spec("keras_tuner") is not None:
    import keras_tuner as kt

    def fine_tune_hyper_parameters(
        model_factory: Callable[[kt.HyperParameters], tf.keras.Model],
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 1,
        max_trials: int = 20,
        prng_seed: int = 42,
        name: str = "experiment",
    ) -> kt.HyperParameters:
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

    def __init__(
        self,
        quantiles: Sequence[int],
        output_size: int = 1,
        **kwargs,
    ):
        """

        Parameters
        ----------
        quantiles:
            Quantiles to compute losses
        output_size:
        kwargs

        """
        super().__init__(**kwargs)
        for quantile in quantiles:
            if quantile < 0 or quantile > 1:
                raise ValueError(
                    f"Illegal quantile value={quantile}! Values should be between 0 and 1."
                )
        self.quantiles = tf.constant(quantiles)
        self.output_size = tf.constant(output_size)
        self.n_quantiles = len(quantiles)

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

        y_true = tf.cast(y_true, y_pred.dtype)
        quantiles = tf.cast(self.quantiles, y_pred.dtype)
        tf.debugging.assert_rank(y_true, 3)

        loss = tf.TensorArray(
            size=self.n_quantiles, dtype=y_pred.dtype, clear_after_read=True
        )

        for i in tf.range(self.n_quantiles):
            indexes = tf.range(i * self.output_size, (i + 1) * self.output_size)
            q_loss = quantile_loss(
                y_true, tf.gather(y_pred, indexes, axis=-1), quantiles[i]
            )
            loss = loss.write(i, q_loss)

        loss = tf.reduce_sum(loss.stack(), axis=0)
        return loss

    def get_config(self) -> Dict[str, ...]:
        config = super().get_config()
        config.update(
            {
                "quantiles": list(self.quantiles),
                "output_size": int(self.output_size),
                "accumulate_in_tensor_array": self.accumulate_in_tensor_array,
            }
        )
        return config


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
    positive_prediction_underflow = tf.maximum(prediction_underflow, 0)
    negative_prediction_underflow = tf.maximum(-prediction_underflow, 0)
    q_loss = tf.add(
        quantile * positive_prediction_underflow,
        (1 - quantile) * negative_prediction_underflow,
    )
    return tf.reduce_sum(q_loss, axis=-2)


default_element_spec = {
    "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
    "time": tf.TensorSpec([None, 192, 1], dtype=tf.float32),
    "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float32),
    "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int32),
    "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float32),
    "inputs_known_categorical": tf.TensorSpec([None, 192, 3], dtype=tf.int32),
    "inputs_observed": tf.TensorSpec([None, 192, 3], dtype=tf.float32),
}


def default_map_fn(
    arg: Mapping[str, tf.Tensor], dtype: DType
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    return (
        dict(
            static=arg["inputs_static"],
            known_real=tf.cast(arg["inputs_known_real"], dtype),
            known_categorical=arg["inputs_known_categorical"],
            observed=tf.cast(arg["inputs_observed"], dtype),
        ),
        tf.cast(arg["outputs"], dtype),
    )


def load_sharded_dataset(
    path: str,
    batch_size: int,
    element_spec: Mapping[str, tf.TensorSpec] | Tuple[tf.TensorSpec, ...] | None = None,
    map_fn: Optional[
        Callable[[Mapping[str, tf.Tensor]], Tuple[Dict[str, tf.Tensor], tf.Tensor]]
    ] = None,
    dtype: DType = tf.float32,
    cache_filename: str = "",
    drop_remainder: bool = False,
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
    drop_remainder:

    Returns
    -------

    retval:
        tf.data.Dataset ready to use for training.

    """

    if element_spec is None:
        element_spec = default_element_spec

    if map_fn is None:
        map_fn = functools.partial(default_map_fn, dtype=dtype)

    return (
        tf.data.Dataset.from_tensor_slices(glob(f"{path}/*"))
        .flat_map(
            lambda i: tf.data.Dataset.load(i, element_spec=element_spec).map(
                map_fn, num_parallel_calls=tf.data.AUTOTUNE
            ),
        )
        .rebatch(batch_size, drop_remainder=drop_remainder)
        .cache(cache_filename)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_data_from_archive(
    path: str,
) -> Dict[str, np.ndarray]:
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


def load_dataset_from_archive(
    path: str,
    batch_size: int,
    element_spec: Mapping[str, tf.TensorSpec] | Tuple[tf.TensorSpec, ...] | None = None,
    map_fn: Optional[
        Callable[[Mapping[str, tf.Tensor]], Tuple[Dict[str, tf.Tensor], tf.Tensor]]
    ] = None,
    dtype: DType = tf.float32,
    cache_filename: str = "",
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    data = load_data_from_archive(path)

    if element_spec is None:
        element_spec = default_element_spec

    if map_fn is None:
        map_fn = functools.partial(default_map_fn, dtype=dtype)

    if drop_remainder:
        batches = list(gen_batches(len(data["identifier"]), batch_size))
        last_element_len = batches[-1].stop - batches[-1].start
        if last_element_len != batch_size:
            batches = batches[:-1]

    def generator():
        for i in batches:
            length = i.stop - i.start
            if length == batch_size:
                yield map_dict(data, value_mapper=lambda v: v[i.start : i.stop])

    return (
        tf.data.Dataset.from_generator(generator, element_spec=element_spec)
        .map(map_fn)
        .cache(cache_filename)
    )
