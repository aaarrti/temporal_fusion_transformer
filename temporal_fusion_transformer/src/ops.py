from __future__ import annotations

import logging
from typing import ContextManager, Sequence

from jax import Array
from keras import backend, ops

log = logging.getLogger(__name__)


def weighted_sum(arr: Array, weights: Array, axis: int | Sequence[int] | None = None) -> Array:
    with named_scope("weighted_sum"):
        return ops.sum(arr * weights, axis=axis)


def named_scope(name: str) -> ContextManager[None, None, None]:
    b = backend.backend()
    if b == "tensorflow":
        import tensorflow as tf

        return tf.name_scope(name)
    elif b == "jax":
        import jax

        return jax.named_scope(name)
    else:
        log.error(f"Unsupported backend {backend.backend()}")
