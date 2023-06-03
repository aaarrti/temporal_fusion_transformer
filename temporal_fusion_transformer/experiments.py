from __future__ import annotations

from abc import ABC, abstractmethod
from enum import auto, IntEnum
from typing import NamedTuple, List

import tensorflow as tf
import polars as pl


class DataType(IntEnum):
    REAL_VALUED = auto()
    CATEGORICAL = auto()
    DATE = auto()


class InputType(IntEnum):
    TARGET = auto()
    OBSERVED_INPUT = auto()
    KNOWN_INPUT = auto()
    STATIC_INPUT = auto()
    # Single column used as an entity identifier
    ID = auto()
    # Single column exclusively used as a time index
    TIME = auto()


class DataConfig(NamedTuple):
    """
    Attributes
    ----------

    static_categories_sizes:
    known_categories_sizes:
    num_encoder_steps:
    num_outputs:



    """

    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    num_outputs: int


class ModelConfig(NamedTuple):
    """
    Attributes
    ----------

    dropout_rate:
    hidden_layer_size:
    num_attention_heads:

    quantiles:
        TODO: WTF are quantiles ????

    """

    dropout_rate: float = 0.1
    hidden_layer_size: int = 5
    num_attention_heads: int = 4

    return_attentions: bool = False
    quantiles = [0.1, 0.5, 0.9]


class Experiment(ABC):
    @property
    @abstractmethod
    def data_config(self) -> DataConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @property
    @abstractmethod
    def train_split(self) -> tf.data.Dataset:
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_split(self) -> tf.data.Dataset:
        raise NotImplementedError


class ElectricityExperiment(Experiment):
    def from_data_frame(self, df: pl.DataFrame) -> ElectricityExperiment:
        """
        - Find real columns -> apply sklearn.NormalScaler
            - real scaler are grouped per ID
            - known and observed
        - find categorical columns -> encode them as integers
            - static and known
        - those both become inputs (TFTInput instance)
        - time ???
        - ids ???
        - targets ???

        Each dataset entry consists of: (i guess)
            - id
            - time
            - outputs (aka TARGET)
            - TFTInput instance

        Parameters
        ----------
        df

        Returns
        -------

        """
        pass
