from __future__ import annotations

import logging
import os
import pathlib
import pickle
from collections import defaultdict
from datetime import date, datetime, timedelta
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import polars as pl
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.datasets.base import (
    MultiHorizonTimeSeriesDataset,
    PreprocessorBase,
    SplitSpec,
)
from temporal_fusion_transformer.src.datasets.utils import (
    time_series_dataset_from_dataframe,
    report_columns_mismatch,
)

_ID_COLUMN = "id"
_REAL_INPUTS = ["year"]
_CATEGORICAL_INPUTS = ["month", "day", "hour", "day_of_week"]
#  Filter to match range used by other academic papers
_CUTTOFF_DAYS = (1096, 1346)
_INPUTS = _REAL_INPUTS + _CATEGORICAL_INPUTS + [_ID_COLUMN]
_TARGETS = ["power_usage"]
log = logging.getLogger(__name__)


class ElectricityDataset(MultiHorizonTimeSeriesDataset):
    def __init__(self, config: Config):
        """
        References
        ----------

        - https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

        Parameters
        ----------

        """
        self.config = config
        self.preprocessor = ElectricityPreprocessor()

    def convert_to_parquet(
        self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True
    ):
        if output_dir is None:
            output_dir = download_dir

        if pathlib.Path(f"{output_dir}/LD2011_2014.parquet").is_file():
            log.info("Found LD2011_2014.parquet, will re-use it.")
            return

        with open(f"{download_dir}/LD2011_2014.txt") as file:
            txt_content = file.read()

        csv_content = txt_content.replace(",", ".").replace(";", ",")

        with TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/LD2011_2014.csv", "w+") as file:
                file.write(csv_content)

            pl.scan_csv(
                f"{tmpdir}/LD2011_2014.csv", infer_schema_length=999999, try_parse_dates=True
            ).rename({"": "timestamp"}).sink_parquet(f"{output_dir}/LD2011_2014.parquet")

        if delete_processed:
            os.remove(f"{download_dir}/LD2011_2014.txt")

    def make_dataset(
        self,
        data_dir: str,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, ElectricityPreprocessor]:
        df = read_parquet(data_dir)

        report_columns_mismatch(df, [_ID_COLUMN] + _INPUTS + _TARGETS)

        self.preprocessor.fit(df)

        split_spec = compute_split_spec(self.config)

        training_df, validation_df, test_df = split_data(df, split_spec)
        training_time_series = self.make_time_series(training_df)
        validation_time_series = self.make_time_series(validation_df)

        return training_time_series, validation_time_series, test_df, self.preprocessor

    def make_time_series(self, df: pl.DataFrame):
        return time_series_dataset_from_dataframe(
            df,
            inputs=_INPUTS,
            targets=_TARGETS,
            id_column=_ID_COLUMN,
            total_time_steps=self.config.total_time_steps,
            preprocessor=self.preprocessor.transform,
        )

    def plot_dataset_splits(self, data_dir: str, entity: str):
        split_spec = compute_split_spec(self.config)

        plt.axvline(x=split_spec.train_end, color="red", linestyle="dashed", label="train end")

        plt.axvline(
            x=split_spec.val_start, color="blue", linestyle="dashed", label="validation start"
        )

        plt.axvline(
            x=split_spec.val_end, color="orange", linestyle="dashed", label="validation end"
        )

        plt.axvline(x=split_spec.test_start, color="green", linestyle="dashed", label="test start")

        df = read_parquet(data_dir).filter(pl.col("id") == entity)

        x = df["timestamp"]
        y = df["power_usage"]

        plt.plot(x, y, color="gray")
        plt.title(entity)
        plt.legend()
        plt.tight_layout()
        plt.show()


class ElectricityPreprocessor(PreprocessorBase):
    def __init__(
        self,
        real: dict[str, StandardScaler] | None = None,
        target: dict[str, StandardScaler] | None = None,
        categorical: dict[str, LabelEncoder] | None = None,
    ):
        if real is None:
            real = defaultdict(StandardScaler)

        if target is None:
            target = defaultdict(StandardScaler)

        if categorical is None:
            categorical = defaultdict(LabelEncoder)

        self.real = real
        self.categorical = categorical
        self.target = target

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        def categorical_mapper(encoder: LabelEncoder) -> pl.Series:
            def map_fn(s: pl.Series) -> pl.Series:
                return pl.Series(encoder.transform(s.to_numpy()))

            return map_fn

        def float_mapper(target: np.ndarray):
            def map_fn(s: pl.Series) -> pl.Series:
                return pl.Series(target).cast(pl.Float32)

            return map_fn

        def group_mapper(group_df: pl.DataFrame) -> pl.DataFrame:
            df_id = group_df["id"][0]

            x_real = group_df[_REAL_INPUTS].to_numpy(order="c")

            x_real = self.real[df_id].transform(x_real)

            group_df = group_df.with_columns(
                [
                    pl.col(j).map_batches(float_mapper(i), pl.Float32)
                    for i, j in zip(x_real.T, _REAL_INPUTS)
                ]
            )

            x_target = group_df[_TARGETS].to_numpy(order="c")
            x_target = self.target[df_id].transform(x_target)
            group_df = group_df.with_columns(
                [
                    pl.col(j).map_batches(float_mapper(i), pl.Float32)
                    for i, j in zip(x_target.T, _TARGETS)
                ]
            )
            return group_df

        df = (
            df.group_by("id")
            .map_groups(group_mapper)
            .with_columns(
                [
                    pl.col(i).map_batches(
                        categorical_mapper(self.categorical[i]), return_dtype=pl.Int64
                    )
                    for i in [_ID_COLUMN] + _CATEGORICAL_INPUTS
                ]
            )
            .shrink_to_fit(in_place=True)
            .rechunk()
        )
        return df

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

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

    @staticmethod
    def load(dirname: str) -> ElectricityPreprocessor:
        real = joblib.load(f"{dirname}/preprocessor.real.joblib")
        target = joblib.load(f"{dirname}/preprocessor.target.joblib")
        categorical = joblib.load(f"{dirname}/preprocessor.categorical.joblib")
        return ElectricityPreprocessor(real=real, target=target, categorical=categorical)

    def fit(self, df: pl.DataFrame):
        for i, sub_df in df.groupby("id"):
            self.target[i].fit(df[_TARGETS].to_numpy(order="c"))
            self.real[i].fit(df[_REAL_INPUTS].to_numpy(order="c"))

        for i in [_ID_COLUMN] + _CATEGORICAL_INPUTS:
            self.categorical[i].fit(df[i].to_numpy())


def read_parquet(data_dir: str) -> pl.DataFrame:
    lf = pl.scan_parquet(f"{data_dir}/LD2011_2014.parquet").collect().pipe(cutoff_df).lazy()

    timeseries_ids = lf.columns[1:]

    # down sample to 1h https://pola-rs.github.io/polars-book/user-guide/transformations/time-series/rolling/
    lf = (
        lf.sort("timestamp")
        .groupby_dynamic("timestamp", every="1h")
        .agg([pl.col(i).mean() for i in timeseries_ids])
    )
    lf_list = []

    for label in timeseries_ids:
        sub_lf = lf.select("timestamp", label)

        sub_lf = sub_lf.rename({label: "power_usage"}).with_columns(
            [
                pl.col("power_usage").cast(pl.Float32),
                pl.col("timestamp").dt.year().alias("year").cast(pl.UInt16),
                pl.col("timestamp").dt.month().alias("month").cast(pl.UInt8),
                pl.col("timestamp").dt.hour().alias("hour").cast(pl.UInt8),
                pl.col("timestamp").dt.day().alias("day").cast(pl.UInt8),
                pl.col("timestamp").dt.weekday().alias("day_of_week").cast(pl.UInt8),
            ],
            id=pl.lit(label),
        )
        lf_list.append(sub_lf)

    df = pl.concat(pl.collect_all(lf_list)).shrink_to_fit(in_place=True).rechunk()
    log.debug(f"{df.null_count() = }")
    return df


def split_data(
    df: pl.DataFrame,
    split_spec: SplitSpec,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    This dataset was recorded in interval [2011-01-01, 2015-01-01].
    """

    train_df = df.filter(pl.col("timestamp") < split_spec.train_end)
    validation_df: pl.DataFrame = df.filter(pl.col("timestamp") >= split_spec.val_start).filter(
        pl.col("timestamp") < split_spec.val_end
    )
    test_df = df.filter(pl.col("timestamp") >= split_spec.test_start)
    return (
        train_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        validation_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        test_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
    )


def cutoff_df(df: pl.DataFrame) -> pl.DataFrame:
    log.debug(f"Before cutoff {len(df) = }")
    start_date: datetime = df["timestamp"].min()
    cutoff_left = start_date + timedelta(days=_CUTTOFF_DAYS[0])
    cutoff_right = start_date + timedelta(days=_CUTTOFF_DAYS[1])
    df = (
        df.lazy()
        .filter(pl.col("timestamp") >= cutoff_left)
        .filter(pl.col("timestamp") <= cutoff_right)
        .collect()
    )
    log.debug(f"After cutoff {len(df) = }")
    return df


def compute_split_spec(config: Config) -> SplitSpec:
    validation_boundary: date = config.validation_boundary
    test_boundary: date = config.test_boundary

    val_df_start = validation_boundary - timedelta(days=config.split_overlap)
    test_df_start = test_boundary - timedelta(days=config.split_overlap)

    return SplitSpec(
        train_end=validation_boundary,
        val_end=test_boundary,
        val_start=val_df_start,
        test_start=test_df_start,
    )
