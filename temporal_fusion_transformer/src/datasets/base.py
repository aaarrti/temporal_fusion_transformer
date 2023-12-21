from __future__ import annotations

import dataclasses
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

DS = TypeVar("DS", bound=dict[str, StandardScaler])
DL = TypeVar("DL", bound=dict[str, LabelEncoder])

if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
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
    def __init__(
        self,
        real: Mapping[str, StandardScaler],
        target: Mapping[str, StandardScaler],
        categorical: Mapping[str, LabelEncoder],
    ):
        self.real = real
        self.target = target
        self.categorical = categorical

    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def save(self, dirname: str):
        real = dict(**self.real)
        target = dict(**self.target)
        categorical = dict(**self.categorical)

        joblib.dump(
            real,
            f"{dirname}/preprocessor.real.joblib",
            compress=3,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        joblib.dump(
            target,
            f"{dirname}/preprocessor.target.joblib",
            compress=3,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        joblib.dump(
            categorical,
            f"{dirname}/preprocessor.categorical.joblib",
            compress=3,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def load(cls, dirname: str) -> PreprocessorBase:
        real = joblib.load(f"{dirname}/preprocessor.real.joblib")
        target = joblib.load(f"{dirname}/preprocessor.target.joblib")
        categorical = joblib.load(f"{dirname}/preprocessor.categorical.joblib")
        return cls(real=real, target=target, categorical=categorical)

    def __str__(self) -> str:
        return f"{type(self).__name__}(real={str(self.real)}, target={str(self.target)}, categorical={str(self.categorical)})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(real={repr(self.real)}, target={repr(self.target)}, categorical={repr(self.categorical)})"
