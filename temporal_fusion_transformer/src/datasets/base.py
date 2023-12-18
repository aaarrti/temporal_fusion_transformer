from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime


class MultiHorizonTimeSeriesDataset(ABC):
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
        self, data_dir: str
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, PreprocessorBase]:
        """
        This method expect data to be in parquet format.

        Parameters
        ----------
        data_dir:
            Directory containing parquet files.

        Returns
        -------

        retval:
            - train split (tf.data.Dataset, ready to use)
            - validation split (tf.data.Dataset, ready to use)
            - test split (pl.Dataframe, must apply preprocessor before passing data to model)
            - experiment specific preprocessor

        """
        raise NotImplementedError

    def plot_dataset_splits(self, data_dir: str, entity: str):
        pass


class PreprocessorBase(ABC):
    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, dirname: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(dirname: str) -> PreprocessorBase:
        raise NotImplementedError
