from __future__ import annotations

from importlib import util
import logging
from contextlib import contextmanager
import platform
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
    Type,
    ContextManager,
)

import numpy as np
import tensorflow as tf
from keras_pbar import keras_pbar
from sklearn.utils import gen_batches
from tensorflow.python.types.core import TensorLike
from tensorflow.python import pywrap_tfe

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.modeling import (
        TemporalFusionTransformer as TF_TemporalFusionTransformer,
    )
    from temporal_fusion_transformer.src.experiments import Experiment

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


def make_tft_model(experiment: Experiment, **kwargs) -> TF_TemporalFusionTransformer:
    """
    Create TFT model for experiment.

    Parameters
    ----------
    experiment:
        Experiment instance used to fill fixed model parameters, as well as default ones.
    kwargs:
        Use this to override default hyperparameters from experiment instance, or to pass additional
        __init__ kwargs to model.

    Returns
    -------

    tft_model:
        TF or Flax implementation of model. In both cases, the model is not traced!
        It is users responsibility to provide representative input.

    """
    kwargs = add_default_items(
        kwargs,
        dict(
            static_categories_sizes=experiment.fixed_params.static_categories_sizes,
            known_categories_sizes=experiment.fixed_params.known_categories_sizes,
            num_encoder_steps=experiment.fixed_params.num_encoder_steps,
            hidden_layer_size=experiment.default_params.hidden_layer_size,
            num_attention_heads=experiment.default_params.num_attention_heads,
            dropout_rate=experiment.default_params.dropout_rate,
        ),
    )
    from temporal_fusion_transformer.src.modeling import (
        TemporalFusionTransformer as TF_TemporalFusionTransformer,
    )

    return TF_TemporalFusionTransformer(**kwargs)


def as_tensor(arr: TensorLike | tf.Tensor) -> tf.Tensor:
    if not isinstance(arr, tf.Tensor):
        return tf.convert_to_tensor(arr)
    else:
        return arr


def assert_quantile_values(quantiles: Sequence[float] | None):
    if quantiles is None:
        return
    for quantile in quantiles:
        if quantile < 0 or quantile > 1:
            raise ValueError(
                f"Illegal quantile value={quantile}! Values should be between 0 and 1."
            )


def add_default_items(
    dictionary: Mapping[str, ...] | None, default_items: Mapping[str, ...]
) -> Dict[str, Any]:
    """Add default_items into dictionary if not present."""
    if dictionary is None:
        return dict(**default_items)
    if len(dictionary) == 0:
        return dict(**default_items)

    copy = dict(**dictionary)

    for k, v in default_items.items():
        if k not in copy:
            copy[k] = v

    return copy


class NoOpStrategy:
    @contextmanager
    def scope(self) -> ContextManager:
        yield


if util.find_spec("flax") is not None:
    from temporal_fusion_transformer.src.modeling_flax import (
        TemporalFusionTransformer as Flax_TemporalFusionTransformer,
    )
    import flax.linen as nn

    def make_flax_tft_model(
        experiment: Experiment, jit: bool = True, **kwargs
    ) -> Flax_TemporalFusionTransformer:
        """
        Create TFT model for experiment.

        Parameters
        ----------
        experiment:
            Experiment instance used to fill fixed model parameters, as well as default ones.
        jit:
        kwargs:
            Use this to override default hyperparameters from experiment instance, or to pass additional
            __init__ kwargs to model.

        Returns
        -------

        tft_model:
            TF or Flax implementation of model. In both cases, the model is not traced!
            It is users responsibility to provide representative input.

        """
        kwargs = add_default_items(
            kwargs,
            dict(
                static_categories_sizes=experiment.fixed_params.static_categories_sizes,
                known_categories_sizes=experiment.fixed_params.known_categories_sizes,
                num_encoder_steps=experiment.fixed_params.num_encoder_steps,
                hidden_layer_size=experiment.default_params.hidden_layer_size,
                num_attention_heads=experiment.default_params.num_attention_heads,
                dropout_rate=experiment.default_params.dropout_rate,
            ),
        )
        clazz = Flax_TemporalFusionTransformer
        if jit:
            clazz = nn.jit(clazz)
        return clazz(**kwargs)


def can_jit_compile(warn=False):
    # Was added only in 2.12.
    """Returns True if TensorFlow XLA is available for the platform."""
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        if warn:
            logging.warning(
                "XLA (`jit_compile`) is not yet supported on Apple M1/M2 ARM "
                "processors. Falling back to `jit_compile=False`."
            )
        return False
    if pywrap_tfe.TF_ListPluggablePhysicalDevices():
        if warn:
            logging.warning(
                "XLA (`jit_compile`) is not supported on your system. "
                "Falling back to `jit_compile=False`."
            )
        return False
    return True
