from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List
from datetime import datetime

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import polars as pl
    import tensorflow as tf
    from typing import Callable

    from temporal_fusion_transformer.src.config_dict import ConfigDict, ModelConfig
    from temporal_fusion_transformer.src.training.metrics import MetricContainer
    from temporal_fusion_transformer.src.config_dict import DataConfig
    from temporal_fusion_transformer.src.training.training_lib import (
        TrainStateContainer,
    )
    from temporal_fusion_transformer.src.modeling.tft_model import TftOutputs

    from temporal_fusion_transformer.src.lib_types import PredictFn, HooksConfig


class MultiHorizonTimeSeriesDataset(ABC):
    @abstractmethod
    def convert_to_parquet(
        self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True
    ):
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
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessorBase] | None:
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

    @classmethod
    def plot_predictions(
        cls, x_batch: np.ndarray, y_batch: np.ndarray, y_predicted: np.ndarray
    ) -> plt.Figure:
        raise NotImplementedError

    @property
    @abstractmethod
    def trainer(self) -> TrainerBase:
        raise NotImplementedError

    @abstractmethod
    def reload_preprocessor(self, filename: str) -> DataPreprocessorBase:
        raise NotImplementedError

    @abstractmethod
    def reload_model(
        self, filename: str, config: ModelConfig, jit_module: bool, return_attention: bool = True
    ) -> PredictFn:
        raise NotImplementedError


class DataPreprocessorBase(ABC):
    @staticmethod
    @abstractmethod
    def load(file_name: str) -> DataPreprocessorBase:
        raise NotImplementedError

    @abstractmethod
    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def convert_dataframe_to_tf_dataset(self, df: pl.DataFrame) -> tf.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x_batch: np.ndarray, y_batch: np.ndarray) -> pl.DataFrame:
        """

        Parameters
        ----------
        x_batch:
            2D batch of inputs passed to model.
        y_batch
            2D batch of model outputs.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def restore_timestamps(self, df: pl.DataFrame) -> List[datetime]:
        """
        Before using this method, inverse_transform must be applied to `df`.

        Parameters
        ----------
        df

        Returns
        -------

        """
        raise NotImplementedError


class TrainerBase(ABC):
    @abstractmethod
    def run(
        self, *args, **kwargs
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError

    @abstractmethod
    def run_distributed(
        self, *args, **kwargs
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError
