from __future__ import annotations


from typing import TYPE_CHECKING, Iterable
from types import SimpleNamespace
from flax.struct import dataclass
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
import optax
import flax.linen as nn
import jax
import jax.numpy as jnp
from orbax.checkpoint import Checkpointer, CheckpointHandler

from keras_pbar import keras_pbar
from temporal_fusion_transformer.flax_.modeling import TemporalFusionTransformer
from absl import logging


if TYPE_CHECKING:
    import tensorflow as tf
    from temporal_fusion_transformer.experiments import Experiment


class SingleDevice(SimpleNamespace):
    pass


class MultiDevice(SimpleNamespace):
    pass


def train_model(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    experiment: Experiment,
):
    devices = jax.devices()
    logging.info(f"{devices = }")
