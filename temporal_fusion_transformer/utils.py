from __future__ import annotations

from typing import (
    Mapping,
    Dict,
    TypeVar,
    Hashable,
    Callable,
    Sequence,
    Tuple,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import tensorflow as tf
from keras_pbar import keras_pbar
from sklearn.utils import gen_batches

if TYPE_CHECKING:
    from temporal_fusion_transformer.tf.modeling import TemporalFusionTransformer
    from temporal_fusion_transformer.experiments import Experiment

T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable, covariant=True)


def map_dict(
    dictionary: Mapping[K, T],
    value_mapper: Callable[[T], R] | None = None,
    *,
    key_mapper: Callable[[K], R] | None = None,
) -> Dict[K | R, V | R]:
    """
    Applies func to values in dict.
    Kinda like tf.nest.map_structure or jax.tree_util.tree_map, but preserves keys.
    Additionally, if provided can also map keys.
    """

    if value_mapper is None:
        value_mapper = identity

    if key_mapper is None:
        key_mapper = identity

    result = {}
    for k, v in dictionary.items():
        new_key = key_mapper(k)
        if isinstance(v, Mapping):
            new_value = map_dict(
                dictionary[k], value_mapper=value_mapper, key_mapper=key_mapper
            )
        else:
            new_value = value_mapper(v)
        result[new_key] = new_value
    return result


def filter_dict(
    dictionary: Mapping[K, V],
    key_filter: Callable[[K], bool] | None = None,
    *,
    value_filter: Callable[[V], bool] | None = None,
) -> Dict[K, V]:
    """
    Filters dictionary based on key or values.
    Kinda like jax.tree_util.tree_filter, but preserves keys.
    """

    def tautology(_) -> bool:
        # Tautology is an expression, which is always true.
        return True

    if key_filter is None:
        key_filter = tautology

    if value_filter is None:
        value_filter = tautology

    result = {}

    for k, v in dictionary.items():
        if key_filter(k) and value_filter(v):
            result[k] = v

    return result


def export_sharded_dataset(
    data: Mapping[str, np.ndarray], export_path: str, shard_size: int = 100_000
):
    """
    Split dataset in shards of size `shard_size`, and write the as TF protobuf to local file system.

    Parameters
    ----------
    data
    export_path
    shard_size

    Returns
    -------

    """
    n = len(data["identifier"])
    batches = gen_batches(n, shard_size)

    n_batches = n // shard_size
    if n % shard_size != 0:
        n_batches += 1

    for index, shard_slice in keras_pbar(enumerate(batches), n_batches):
        shard = map_dict(
            data, value_mapper=lambda v: v[shard_slice.start : shard_slice.stop]
        )
        tf.data.Dataset.from_tensors(shard).save(f"{export_path}/{index}")


def load_sharded_dataset(
    file_names: Sequence[str],
    element_spec: Mapping[str, tf.TensorSpec] | Tuple[tf.TensorSpec, ...] | None = None,
) -> tf.data.Dataset:
    if element_spec is None:
        element_spec = {
            "identifier": tf.TensorSpec([None, 192, 1], dtype=tf.string),
            "time": tf.TensorSpec([None, 192, 1], dtype=tf.float32),
            "outputs": tf.TensorSpec([None, 24, 1], dtype=tf.float32),
            "inputs_static": tf.TensorSpec([None, 1], dtype=tf.int32),
            "inputs_known_real": tf.TensorSpec([None, 192, 3], dtype=tf.float32),
            "inputs_known_categorical": tf.TensorSpec([None, 192, 3], dtype=tf.int32),
            "inputs_observed": tf.TensorSpec([None, 192, 3], dtype=tf.float32),
        }

    return tf.data.Dataset.from_tensor_slices(file_names).flat_map(
        lambda i: tf.data.Dataset.load(i, element_spec=element_spec)
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


def flatten_dict(xs: Mapping[str, ...], sep: str = "/") -> Dict[str, ...]:
    """
    Examples:

    >>> flatten_dict({'a': 1, 'b': {'c': {'d': 4}}})
    >>> {'a': 1, 'b/c/d': 4}
    Parameters
    ----------
    xs
    sep

    Returns
    -------

    """

    def _key(path: Tuple[str, ...]) -> str:
        return sep.join(path)

    def _flatten(xs_i: Any, prefix: Tuple[str, ...]) -> Dict[str, ...]:
        if not isinstance(xs_i, Mapping):
            return {_key(prefix): xs_i}
        result = {}
        for key, value in xs_i.items():
            path = prefix + (key,)
            result.update(_flatten(value, path))
        return result

    return _flatten(xs, ())


def unflatten_dict(xs: Mapping[str, ...], sep: str = "/") -> Dict[str, ...]:
    """
    Examples:
    >>> unflatten_dict({'a': 1, 'b/c/d': 4})
    >>> {'a': 1, 'b': {'c': {'d': 4}}}
    Parameters
    ----------
    xs
    sep

    Returns
    -------

    """
    result = {}
    for path, value in xs.items():
        path = path.split(sep)
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result


def identity(x: T) -> T:
    return x


def assert_rank(x, rank, **kwargs):
    tf.debugging.assert_rank(x, rank, **kwargs)


def make_tft_model(experiment: Experiment, **kwargs) -> TemporalFusionTransformer:
    from temporal_fusion_transformer.tf.modeling import TemporalFusionTransformer

    return TemporalFusionTransformer(
        static_categories_sizes=experiment.fixed_params.static_categories_sizes,
        known_categories_sizes=experiment.fixed_params.known_categories_sizes,
        num_encoder_steps=experiment.fixed_params.num_encoder_steps,
        hidden_layer_size=experiment.default_params.hidden_layer_size,
        num_attention_heads=experiment.default_params.num_attention_heads,
        dropout_rate=experiment.default_params.dropout_rate,
        **kwargs,
    )
