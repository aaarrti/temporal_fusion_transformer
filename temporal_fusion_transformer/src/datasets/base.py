from __future__ import annotations

import abc
from typing import TypeVar, Tuple, Dict, List, ClassVar, OrderedDict, Sequence
import numpy as np
import polars as pl
import tensorflow as tf
from functools import cached_property
from keras.utils import FeatureSpace, timeseries_dataset_from_array
from toolz import functoolz
from absl import logging
from ordered_set import OrderedSet
from tqdm.auto import tqdm

T = TypeVar("T")

"""
In the end, wee want n time series stacked over batch axis.
"""

Triple = Tuple[T, T, T]


class MultiHorizonTimeSeriesDataset(abc.ABC):

    """

    Attributes
    ----------



    """

    target_feature_names: ClassVar[Sequence[str]]
    features: OrderedDict[str, FeatureSpace.Feature]
    total_time_steps: ClassVar[int]
    feature_space: FeatureSpace = property(lambda self: FeatureSpace(self.features, output_mode="dict"))

    def make_dataset(self, path: str) -> Tuple[Triple[tf.data.Dataset], FeatureSpace]:
        """
        Our final goal, are 3 datasets, which yield tensors with shape [batch, time step, value].
        What we need to do is:
        - Read raw data
        - Train feature pre-processor (keras.FeatureSpace)
        - Do a train/validation/test split -> remove all non Feature space columns.
        - Encode categorical features + Normalize real valued features (FeatureSpace.__call__).
        - Create a time-series dataset for each entity (df.group_by).
        - split inputs and targets.
        - Create a 2D grid of ids, so later we can identify to which entity each data-point belongs to.
        - Join into 1 dataset, persist it.

        Parameters
        ----------
        path:
            Path to directory containing CSV files.

        Returns
        -------

        """

        self.download_data(path)
        df = self.read_csv(path)

        nulls = df.select([pl.col(i).null_count() for i in df.columns])

        for col in df.columns:
            n_nulls = nulls[col][0]
            if n_nulls > 0:
                logging.error(nulls)
                raise ValueError(f"Column {col} has {n_nulls} nulls")

        if "id" not in df.columns:
            raise ValueError(f"DataFrame must have `id` column.")

        s1 = set(df.columns)
        s2 = set(self.feature_space.features.keys())
        if s1 != s2:
            raise ValueError(
                f"Expected dataframe to have same columns as feature_space, but found: "
                f"dataframe.columns = {s1}, "
                f"features_space.keys = {s2}, "
                f"mismatches -> {s1.symmetric_difference(s2)}"
            )
        training_df, validation_df, test_df = self.split_data(df)

        # FeatureSpace accepts only tf.data.Dataset
        data_dict = {k: v.to_numpy() for k, v in df.to_dict().items()}
        tf_ds = tf.data.Dataset.from_tensors(data_dict)
        feature_space = self.feature_space
        feature_space.adapt(tf_ds)

        del tf_ds
        del df
        logging.info("Creating training time-series dataset.")
        training_ds = self._make_time_series_dataset(training_df, feature_space)
        logging.info("Creating validation time-series dataset.")
        validation_ds = self._make_time_series_dataset(validation_df, feature_space)
        logging.info("Creating test time-series dataset.")
        test_ds = self._make_time_series_dataset(test_df, feature_space)
        return (training_ds, validation_ds, test_ds), feature_space

    @cached_property
    def _input_feature_names(self) -> List[str]:
        features_from_feature_space = OrderedSet(list(self.features.keys()))
        input_features = features_from_feature_space.symmetric_difference(OrderedSet(list(self.target_feature_names)))
        return list(input_features)

    def _unpack_x_y(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        data = tf.nest.map_structure(np.squeeze, data)

        y = [data.pop(i) for i in self.target_feature_names]
        x = [data.pop(i) for i in self._input_feature_names]

        x = tf.stack(x, axis=1)
        y = tf.stack(y, axis=1)

        return x, y

    def _make_time_series_dataset(
        self, df: pl.DataFrame | pl.LazyFrame, feature_space: FeatureSpace
    ) -> tf.data.Dataset:
        if isinstance(df, pl.LazyFrame):
            df: pl.DataFrame = df.collect()

        time_series_list = []
        number_of_groups = len(list(df.groupby("id")))

        for id_i, df_i in tqdm(df.groupby("id"), total=number_of_groups, desc="Converting to time-series dataset"):
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

        return functoolz.reduce(lambda a, b: a.concatenate(b), time_series_list)

    @abc.abstractmethod
    def download_data(self, path: str):
        """Download raw data into `path` directory."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_csv(self, path: str) -> pl.DataFrame | pl.LazyFrame:
        """Read raw data from `path` directory, do necessary preprocessing."""
        raise NotImplementedError

    @abc.abstractmethod
    def split_data(self, df: pl.DataFrame | pl.LazyFrame) -> Triple[pl.DataFrame | pl.LazyFrame]:
        """Split data into training/validation/test."""
        raise NotImplementedError

    @staticmethod
    def integer_categorical() -> FeatureSpace.Feature:
        """Just a shortcut to avoid typing the same long types."""
        return FeatureSpace.integer_categorical(num_oov_indices=0, output_mode="int")

    @staticmethod
    def string_categorical() -> FeatureSpace.Feature:
        """Just a shortcut to avoid typing the same long types."""
        return FeatureSpace.string_categorical(num_oov_indices=0, output_mode="int")
