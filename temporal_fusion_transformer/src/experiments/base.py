from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, overload

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import ConfigDict
    from temporal_fusion_transformer.src.training.metrics import MetricContainer
    from temporal_fusion_transformer.src.training.training_lib import (
        TrainStateContainer,
    )


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

    @overload
    def make_dataset(
        self, data_dir: str, save_dir: str
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessorBase]:
        ...

    @overload
    def make_dataset(self, data_dir: str, save_dir: None = None) -> None:
        ...

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
    def inverse_transform(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class TrainerBase(ABC):
    @abstractmethod
    def run(
        self,
        data_dir: str,
        batch_size: int,
        config: ConfigDict,
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        verbose: bool = True,
        profile: bool = False,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError

    @abstractmethod
    def run_distributed(
        self, *args, **kwargs
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError
