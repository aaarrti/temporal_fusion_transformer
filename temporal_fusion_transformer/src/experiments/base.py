from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Tuple

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
    def make_dataset(
        self, data_dir: str, mode: Literal["persist", "return"]
    ) -> None | Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessorBase]:
        raise NotImplementedError

    @classmethod
    def plot_predictions(cls, x_batch: np.ndarray, y_batch: np.ndarray, y_predicted: np.ndarray) -> plt.Figure:
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
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
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
        save_path: str | None = None,
        verbose: bool = True,
        profile: bool = False,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError

    @abstractmethod
    def run_distributed(
        self,
        data_dir: str,
        batch_size: int,
        config: ConfigDict,
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        save_path: str | None = None,
        verbose: bool = True,
        profile: bool = False,
        device_type: Literal["gpu", "tpu"] = "gpu",
        prefetch_buffer_size: int = 2,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        raise NotImplementedError
