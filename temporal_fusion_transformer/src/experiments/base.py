from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Literal, Tuple, Type, TypedDict

import keras_core
import keras_core as keras
import numpy as np
from keras_core import layers
from keras_core.src.saving import serialization_lib
from tree import map_structure

from temporal_fusion_transformer.src.utils.utils import classproperty

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import polars as pl
    import tensorflow as tf


class Experiment:
    @classproperty
    def dataset(self) -> Type[MultiHorizonTimeSeriesDataset]:
        raise NotImplementedError

    @classproperty
    def preprocessor(self) -> Type[Preprocessor]:
        raise NotImplementedError

    @staticmethod
    def train_model(
        data_dir: str = "data", batch_size=128, epochs: int = 1, save_filename: str | None = "model.keras", **kwargs
    ) -> keras.Model | None:
        raise NotImplementedError

    @staticmethod
    def train_model_distributed(*args, **kwargs):
        raise NotImplementedError


class MultiHorizonTimeSeriesDataset(ABC):
    @abstractmethod
    def convert_to_parquet(self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True):
        """
        Convert data to parquet format to reduce memory requirement.

        Parameters
        ----------
        download_dir:
            Directory with downloaded CSV files.
        output_dir
            Directory, in which .parquet file must be written
        delete_processed:


        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def make_dataset(
        self, data_dir: str, save_dir: str | None = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, Preprocessor] | None:
        """
        This method expect data to be in parquet format.

        Parameters
        ----------
        data_dir:
            Directory containing parquet files.
        save_dir:
            If not None, instead of returning data splits, will persist them to  `save_dir`

        Returns
        -------

        retval:
            - train split (tf.data.Dataset, ready to use)
            - validation split (tf.data.Dataset, ready to use)
            - test split (pl.Dataframe, must apply preprocessor before passing data to model)
            - experiment specific preprocessor

        """
        raise NotImplementedError

    # @abstractmethod
    def plot_predictions(
        self,
        df: pl.DataFrame,
        entity: str,
        preprocessor,
        model,
        batch_size: int = 32,
        truncate_past: datetime | None = None,
    ) -> plt.Figure:
        raise NotImplementedError

    # @abstractmethod
    def plot_feature_importance(
        self,
        df: pl.DataFrame,
        entity: str,
        preprocessor,
        model,
        batch_size: int = 32,
        truncate_past: datetime | None = None,
    ) -> plt.Figure:
        raise NotImplementedError


if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf


class PreprocessorState(TypedDict):
    real: Dict[str, layers.Normalization]
    target: Dict[str, layers.Normalization]
    categorical: Dict[str, layers.IntegerLookup | layers.StringLookup]


class Preprocessor(keras_core.Model):
    state: PreprocessorState

    def __init__(self, state: PreprocessorState, name="preprocessor", **kwargs):
        super().__init__(name=name, **kwargs)
        self.state = state
        self.run_eagerly = True

    def __call__(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return self.apply(df)

    def adapt(self, df: pl.DataFrame):
        pass

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def transform_one(self, arr: np.ndarray, kind: Literal["real", "target", "categorical"], key: str) -> np.ndarray:
        return np.asarray(self.state[kind][key](arr))

    def get_config(self):
        config = super().get_config()
        state = map_structure(serialization_lib.serialize_keras_object, self.state)
        config["state"] = state
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["state"] = {
            "real": {k: serialization_lib.deserialize_keras_object(v) for k, v in config["state"]["real"].items()},
            "target": {k: serialization_lib.deserialize_keras_object(v) for k, v in config["state"]["target"].items()},
            "categorical": {
                k: serialization_lib.deserialize_keras_object(v) for k, v in config["state"]["categorical"].items()
            },
        }

        model = super().from_config(config, custom_objects)
        model.built = True
        return model
