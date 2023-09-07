from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, TypedDict

import numpy as np
import tensorflow as tf
from absl_extra.flax_utils import save_as_msgpack
from flax.serialization import msgpack_restore
from jax.tree_util import tree_map
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import LabelEncoder, StandardScaler

if TYPE_CHECKING:
    import polars as pl


class PreprocessorDict(TypedDict):
    real: Any
    target: Any
    categorical: Any


class StandardScalerPytree(TypedDict):
    var: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


class LabelEncoderPytree(TypedDict):
    classes: np.ndarray


def standard_scaler_to_pytree(sc: StandardScaler) -> StandardScalerPytree:
    return {"var": sc.var_, "mean": sc.mean_, "scale": sc.scale_}


def pytree_to_standard_scaler(pytree: StandardScalerPytree) -> StandardScaler:
    sc = StandardScaler()
    sc.var_ = pytree["var"]
    sc.mean_ = pytree["mean"]
    sc.scale_ = pytree["scale"]
    return sc


def label_encoder_to_pytree(le: LabelEncoder) -> LabelEncoderPytree:
    classes = le.classes_
    if isinstance(classes, np.ndarray) and classes.dtype == object:
        classes = classes.tolist()

    return {"classes": classes}


def pytree_to_label_encoder(pytree: LabelEncoderPytree) -> LabelEncoder:
    le = LabelEncoder()
    le.classes_ = pytree["classes"]
    return le


def serialize_preprocessor(
    preprocessor: PreprocessorDict,
    data_dir: str,
):
    def is_leaf(sc):
        return isinstance(sc, (StandardScaler, LabelEncoder))

    def map_fn(x):
        if isinstance(x, StandardScaler):
            return standard_scaler_to_pytree(x)
        else:
            return label_encoder_to_pytree(x)

    pytree = tree_map(map_fn, preprocessor, is_leaf=is_leaf)
    save_as_msgpack(pytree, f"{data_dir}/preprocessor.msgpack")


def deserialize_preprocessor(data_dir: str) -> PreprocessorDict:
    with open(f"{data_dir}/preprocessor.msgpack", "rb") as file:
        byte_date = file.read()

    restored = msgpack_restore(byte_date)

    def map_fn(x):
        if isinstance(x, StandardScalerPytree):
            return pytree_to_standard_scaler(x)
        else:
            return label_encoder_to_pytree(x)

    def is_leaf(x):
        return isinstance(x, (LabelEncoderPytree, StandardScalerPytree))

    preprocessor = tree_map(map_fn, restored, is_leaf=is_leaf)
    return preprocessor


def time_series_from_array(
    df: pl.DataFrame, inputs: List[str], targets: List[str], total_time_steps: int
) -> tf.data.Dataset:
    x: np.ndarray = df[inputs + targets].to_numpy(order="c")

    # -1 for TARGETS
    num_inputs = len(inputs)

    time_series: tf.data.Dataset = timeseries_dataset_from_array(
        x,
        None,
        total_time_steps,
        batch_size=None,
    )
    time_series = time_series.map(
        lambda i: (i[..., :num_inputs], i[..., num_inputs:]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    return time_series
