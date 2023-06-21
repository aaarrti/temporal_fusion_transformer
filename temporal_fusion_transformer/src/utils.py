from __future__ import annotations

import logging
import platform
from functools import lru_cache
from contextlib import contextmanager
from importlib import util
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
    ContextManager,
)

import tensorflow as tf
from tensorflow.python import pywrap_tfe
from tensorflow.python.types.core import TensorLike

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


def can_jit_compile(warn=True):
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


def can_use_cudnn() -> bool:
    """
    We can use CuDNN if:
    - gpu available
    - gpu device is cuda
    - tensorflow was able to find CuDNN version
    - TODO: mb we also need to check installation location for actual dll?
    Returns
    -------

    """
    sysconfig: Dict[str, bool] = tf.sysconfig.get_build_info()
    return (
        len(tf.config.list_physical_devices("GPU")) > 0
        and sysconfig["is_cuda_build"]
        and "cudnn_version" in sysconfig
    )


def setup_logging():
    import logging
    import tensorflow as tf
    import absl.logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    )
    tf.get_logger().setLevel("DEBUG")
    absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)


@lru_cache(maxsize=None)
def supports_mixed_precision() -> bool:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    n_gpus = len(gpus)
    if n_gpus == 0:
        return False
    if n_gpus >= 2:
        absl.logging.error(
            f"supports_mixed_precision() check supports only 1 GPU, but found {n_gpus}"
        )

    details = tf.config.experimental.get_device_details(gpus[0])
    cc = details.get("compute_capability")
    return cc is not None and cc >= (7, 0)


@lru_cache
def can_jit_compile() -> bool:
    """Returns True if TensorFlow XLA is available for the platform."""
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        logging.warning(
            "XLA (`jit_compile`) is not yet supported on Apple M1/M2 ARM "
            "processors. Falling back to `jit_compile=False`."
        )
        return False
    if pywrap_tfe.TF_ListPluggablePhysicalDevices():
        logging.warning(
            "XLA (`jit_compile`) is not supported on your system. "
            "Falling back to `jit_compile=False`."
        )

    return True
