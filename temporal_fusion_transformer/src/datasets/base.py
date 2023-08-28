from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Dict, List, Sequence, Tuple, TypeVar
from collections import OrderedDict

import numpy as np
import polars as pl
import tensorflow as tf
from absl import logging
from keras.utils import FeatureSpace, timeseries_dataset_from_array
from toolz import functoolz
from tqdm.auto import tqdm
import gc

T = TypeVar("T")
DF = TypeVar("DF", pl.DataFrame, pl.LazyFrame)
Triple = Tuple[T, T, T]


class MultiHorizonTimeSeriesDataset(ABC):

    """

    Attributes
    ----------


    target_feature_names:
        TODO
    features:
        TODO
    total_time_steps:
        TODO


    """

    target_feature_names: ClassVar[Sequence[str]]
    features: OrderedDict[str, FeatureSpace.Feature]
    total_time_steps: ClassVar[int]
    id_column: ClassVar[str] = "id"

    @cached_property
    def feature_space(self) -> FeatureSpace:
        features = OrderedDict([(i, FeatureSpace.float_normalized()) for i in self.target_feature_names])
        features.update(self.features)
        return FeatureSpace(features, output_mode="dict")

    @cached_property
    def input_feature_names(self) -> List[str]:
        return [i for i in self.features.keys() if i not in self.target_feature_names]

    def make_dataset(self, path: str) -> Tuple[Triple[tf.data.Dataset], FeatureSpace]:
        """
        Our final goal, are 3 datasets, which yield tensors with shape [batch, time step, value].
        What we need to do is:
        - Read raw data
        - Train feature pre-processor (keras.FeatureSpace)
        - Do a train/validation/test split -> remove all non Feature space columns.
        - Encode categorical features + Normalize real valued features (FeatureSpace.__call__).
        - Create a time-series dataset for each entity (lf.group_by).
        - split inputs and targets.
        - Create a 2D grid of ids, so later we can identify to which entity each data-point belongs to.
        - Join into 1 dataset, persist it.

        In the end, wee want n time series stacked over batch axis.

        Parameters
        ----------
        path:
            Path to directory containing CSV files.

        Returns
        -------

        """
        if self.needs_download(path):
            self.download_data(path)
            self.convert_to_parquet(path)

        df = self.read_parquet(path)

        nulls = df.select([pl.col(i).null_count() for i in df.columns])

        for col in df.columns:
            n_nulls = nulls[col][0]
            if n_nulls > 0:
                logging.error(nulls)
                raise ValueError(f"Column {col} has {n_nulls} nulls")

        if self.id_column not in df.columns:
            raise ValueError(f"DataFrame must have `id` column.")

        s1 = set(df.columns)
        s2 = set(self.feature_space.features.keys())
        if s1 != s2:
            logging.error(
                f"Expected dataframe to have same columns as feature_space, but found: "
                f"dataframe.columns = {s1}, "
                f"features_space.keys = {s2}, "
                f"mismatches -> {s1.symmetric_difference(s2)}"
            )

        # FeatureSpace accepts only tf.data.Dataset
        data_dict = {k: v.to_numpy() for k, v in df.to_dict().items()}
        tf_ds = tf.data.Dataset.from_tensors(data_dict)
        feature_space = self.feature_space
        feature_space.adapt(tf_ds)
        del tf_ds
        gc.collect()
        training_df, validation_df, test_df = self.split_data(df)
        del df
        gc.collect()

        logging.info("Creating training time-series dataset.")
        training_ds = self._make_time_series_dataset(training_df, feature_space)
        logging.info("Creating validation time-series dataset.")
        validation_ds = self._make_time_series_dataset(validation_df, feature_space)
        logging.info("Creating test time-series dataset.")
        test_ds = self._make_time_series_dataset(test_df, feature_space)
        return (training_ds, validation_ds, test_ds), feature_space

    def _unpack_x_y(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        data = tf.nest.map_structure(np.squeeze, data)

        y = [data.pop(i) for i in self.target_feature_names]
        x = [data.pop(i) for i in self.input_feature_names]

        x = tf.stack(x, axis=1)
        y = tf.stack(y, axis=1)

        return x, y

    def _make_time_series_dataset(self, df: pl.DataFrame, feature_space: FeatureSpace) -> tf.data.Dataset:
        time_series_list = []
        groups = list(df.partition_by(self.id_column))

        for id_i, df_i in tqdm(groups, desc="Converting to time-series dataset"):
            data_dict_i = {k: list(v.to_numpy()) for k, v in df_i.to_dict().items()}
            data_dict_i = feature_space(data_dict_i)
            x_i, y_i = self._unpack_x_y(data_dict_i)
            # for some reason, keras would generate targets of shape [1, n] and inputs [time_steps, n],
            # but we need time-steps for y_batch also, we need is [time_steps, m]. We don't need `sequence_stride`,
            # since we don't want any synthetic repetitions.
            num_inputs = x_i.shape[-1]
            ts = tf.concat([tf.cast(x_i, tf.float32), y_i], axis=-1)
            time_series: tf.data.Dataset = timeseries_dataset_from_array(
                ts,
                None,
                self.total_time_steps,
                batch_size=None,
            )
            time_series = time_series.map(
                lambda x: (x[..., :num_inputs], x[..., num_inputs:]),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
            )
            time_series_list.append(time_series)

        ds = functoolz.reduce(lambda a, b: a.concatenate(b), time_series_list)
        gc.collect()
        return ds

    @staticmethod
    def integer_categorical() -> FeatureSpace.Feature:
        """Just a shortcut to avoid typing the same long types."""
        return FeatureSpace.integer_categorical(num_oov_indices=0, output_mode="int")

    @staticmethod
    def string_categorical() -> FeatureSpace.Feature:
        """Just a shortcut to avoid typing the same long types."""
        return FeatureSpace.string_categorical(num_oov_indices=0, output_mode="int")

    @abstractmethod
    def needs_download(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def download_data(self, path: str):
        """Download raw data into `path` directory."""
        raise NotImplementedError

    @abstractmethod
    def convert_to_parquet(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read raw data from `path` directory, do necessary preprocessing."""
        raise NotImplementedError

    @abstractmethod
    def split_data(self, df: pl.DataFrame) -> Triple[pl.DataFrame]:
        """Split data into training/validation/test."""
        raise NotImplementedError
