from __future__ import annotations

from typing import ContextManager

from keras import backend


def named_scope(name: str) -> ContextManager:
    if backend.backend() == "tensorflow":
        import tensorflow as tf

        return tf.name_scope(name)

    if backend.backend() == "jax":
        import jax

        return jax.named_scope(name)

    raise RuntimeError(f"Unsupported backend {backend.backend}")
