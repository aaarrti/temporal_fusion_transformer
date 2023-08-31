from __future__ import annotations
from typing import TypedDict, Mapping
import numpy as np
from jax.tree_util import tree_map
from absl_extra.flax_utils import save_as_msgpack

from sklearn.preprocessing import StandardScaler, LabelEncoder


class PreprocessorDict(TypedDict):
    real: Mapping[str, StandardScaler]
    target: Mapping[str, StandardScaler]
    categorical: Mapping[str, LabelEncoder]
    

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
    return {"classes": le.classes_}


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
