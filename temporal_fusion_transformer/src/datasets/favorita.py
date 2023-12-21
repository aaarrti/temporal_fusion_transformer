from __future__ import annotations

import gc
import logging
import os
import pathlib
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob
from importlib import util
from pathlib import Path

import polars as pl
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import _is_fitted  # noqa
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.datasets.base import (
    MultiHorizonTimeSeriesDataset,
    PreprocessorBase,
)
from temporal_fusion_transformer.src.datasets.utils import (
    report_columns_mismatch,
    time_series_dataset_from_dataframe,
)

log = logging.getLogger(__name__)
# date is in interval [2013-1-1, 2017-8-15]
_START_DATE = datetime(2015, 1, 1)
_END_DATE = datetime(2016, 6, 1)
_NUM_IDS = 143658

_REQUIRED_FILES = [
    "stores.parquet",
    "items.parquet",
    "transactions.parquet",
    "oil.parquet",
    "holidays_events.parquet",
]
_REAL_INPUTS = [
    # observed
    "oil_price",
    "transactions",
    # known
    "year",
]
_CATEGORICAL_INPUTS = [
    # static
    "item_nbr",
    "store_nbr",
    "city",
    "state",
    "type",
    "cluster",
    "family",
    "class",
    # known
    "month",
    "day_of_month",
    "day_of_week",
    "national_hol",
    "regional_hol",
    "local_hol",
    "onpromotion",
    "open",
    "perishable",
]
_TARGETS = ["log_sales"]
_ID_COLUMN = "traj_id"


class FavoritaDataset(MultiHorizonTimeSeriesDataset):
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = FavoritaPreprocessor(
            defaultdict(StandardScaler), defaultdict(StandardScaler), defaultdict(LabelEncoder)
        )

    def make_dataset(
        self, data_dir: str, save_dir: str | None = None
    ) -> None | tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, FavoritaPreprocessor]:
        df = read_parquet(data_dir)

        report_columns_mismatch(df, [_ID_COLUMN] + _TARGETS + _REAL_INPUTS + _CATEGORICAL_INPUTS)

        reloaded_preprocessor = maybe_reload_preprocessor(data_dir)

        if reloaded_preprocessor is None:
            self.preprocessor.fit(df)
            self.preprocessor.save(data_dir)
        else:
            self.preprocessor = reloaded_preprocessor

        df = self.preprocessor.transform(df)
        training_df, validation_df, test_df = split_data(df, self.config)

        training_ds = self.make_timeseries(training_df)
        validation_ds = self.make_timeseries(validation_df)

        return training_ds, validation_ds, test_df, self.preprocessor

    def make_timeseries(self, df: pl.DataFrame):
        return time_series_dataset_from_dataframe(
            df,
            inputs=_CATEGORICAL_INPUTS + _REAL_INPUTS,
            targets=_TARGETS,
            id_column="traj_id",
            total_time_steps=self.config.total_time_steps,
        )

    def convert_to_parquet(
        self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True
    ):
        if all([pathlib.Path(f"{download_dir}/{i}").is_file() for i in _REQUIRED_FILES]):
            log.debug(f"Found {_REQUIRED_FILES} locally, will re-use them")
            return

        if output_dir is None:
            output_dir = download_dir

        files = glob(f"{download_dir}/*.csv")
        for file in tqdm(files, desc="Converting to parquet"):
            file: str
            target_file = file.replace(download_dir, output_dir).replace("csv", "parquet")
            pl.scan_csv(file, try_parse_dates=True).sink_parquet(target_file)
            if delete_processed:
                os.remove(file)

    def plot_dataset_splits(self, data_dir: str, entity: str):
        df = (
            pl.scan_parquet(f"{data_dir}/train.parquet")
            .filter(pl.col("traj_id") == entity)
            .collect()
        )

        df = restore_timestamp(df)

        validation_boundary, test_boundary = compute_split_spec(self.config)

        plt.axvline(
            x=validation_boundary, color="green", linestyle="dashed", label="validation boundary"
        )
        plt.axvline(x=test_boundary, color="red", linestyle="dashed", label="test boundary")

        x = df["timestamp"]
        y = df["log_sales"]

        plt.plot(x, y, color="gray", label="log_sales")
        plt.title(entity)
        plt.legend()
        plt.tight_layout()
        plt.show()


class FavoritaPreprocessor(PreprocessorBase):
    def fit(self, df: pl.DataFrame):
        # In contrast for favorita we don't group before training StandardScaler
        for i in _TARGETS:
            self.target[i].fit(df[i].to_numpy().reshape(-1, 1))

        for i in _REAL_INPUTS:
            self.real[i].fit(df[i].to_numpy().reshape(-1, 1))

        for i in _CATEGORICAL_INPUTS:
            self.categorical[i].fit(df[i].to_numpy())

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        lf = df.lazy()

        for i in _TARGETS:
            x = df[i].to_numpy().reshape(-1, 1)
            x = self.target[i].transform(x)
            lf = lf.drop(i).with_columns(pl.lit(x.reshape(-1)).alias(i))

        for i in _REAL_INPUTS:
            x = df[i].to_numpy().reshape(-1, 1)
            x = self.real[i].transform(x)
            lf = lf.drop(i).with_columns(pl.lit(x.reshape(-1)).alias(i))

        for i in _CATEGORICAL_INPUTS:
            x = df[i].to_numpy()
            x = self.categorical[i].transform(x)
            lf = lf.drop(i).with_columns(pl.lit(x).alias(i))

        df = lf.collect().shrink_to_fit(in_place=True).rechunk()
        return df

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @property
    def is_fitted(self) -> bool:
        return (
            all([_is_fitted(v) for v in self.real.values()])
            and all([_is_fitted(v) for v in self.target.values()])
            and all([_is_fitted(v) for v in self.categorical.values()])
        )


if util.find_spec("polars"):
    _COLUMN_TO_DTYPE = {
        "store_nbr": pl.UInt8,
        "cluster": pl.UInt8,
        "perishable": pl.UInt8,
        "class": pl.UInt16,
        "transactions": pl.UInt16,
        "oil_price": pl.Float32,
        "onpromotion": pl.UInt8,
        "item_nbr": pl.UInt32,
        "unit_sales": pl.Float32,
    }


def split_data(
    df: pl.DataFrame,
    config: Config,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    lf = df.lazy()

    validation_boundary, test_boundary = compute_split_spec(config)

    training_df: pl.DataFrame = lf.filter(pl.col("date").lt(validation_boundary)).collect()
    validation_df = df.filter(pl.col("date").ge(validation_boundary)).filter(
        pl.col("date").lt(test_boundary)
    )
    test_df = df.filter(pl.col("date").ge(test_boundary))

    # Filter out identifiers not present in training (i.e. cold-started items).
    identifiers = training_df["traj_id"].unique().to_list()
    ids = set(identifiers)

    def filter_ids(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("traj_id").is_in(ids))

    validation_df = validation_df.pipe(filter_ids)
    test_df = test_df.pipe(filter_ids)

    return training_df, validation_df, test_df


def read_parquet(
    data_dir: str,
    cache_dir: str | None = None,
) -> pl.DataFrame:
    if cache_dir is None:
        cache_dir = data_dir

    if Path(f"{cache_dir}/joined_df.parquet").exists():
        logging.info("Found joined_df.parquet, will re-use it.")
        return pl.read_parquet(f"{cache_dir}/joined_df.parquet")

    temporal = read_temporal(data_dir)
    tmp = temporal.collect(streaming=True)

    store_info = pl.scan_parquet(f"{data_dir}/stores.parquet").pipe(downcast_dataframe)
    items = pl.scan_parquet(f"{data_dir}/items.parquet").pipe(downcast_dataframe)
    transactions = pl.scan_parquet(f"{data_dir}/transactions.parquet").pipe(downcast_dataframe)
    oil = (
        pl.scan_parquet(f"{data_dir}/oil.parquet")
        .rename({"dcoilwtico": "oil_price"})
        .pipe(downcast_dataframe)
    )
    holidays = pl.scan_parquet(f"{data_dir}/holidays_events.parquet")

    national_holidays = (
        holidays.filter(pl.col("locale") == "National")
        .select(["description", "date"])
        .rename({"description": "national_hol"})
        .pipe(downcast_dataframe)
    )
    regional_holidays = (
        holidays.filter(pl.col("locale") == "Regional")
        .select(["description", "locale_name", "date"])
        .rename({"locale_name": "state", "description": "regional_hol"})
        .pipe(downcast_dataframe)
    )
    local_holidays = (
        holidays.filter(pl.col("locale") == "Local")
        .select(["description", "locale_name", "date"])
        .rename({"locale_name": "city", "description": "local_hol"})
        .pipe(downcast_dataframe)
    )

    logging.debug(f"Pre join = {tmp}")
    del tmp
    gc.collect()
    logging.debug("Joining tables")

    df: pl.DataFrame = (
        temporal.join(oil, on="date", how="left")
        .with_columns(pl.col("oil_price").fill_null(strategy="forward"))
        .join(store_info, on="store_nbr")
        .join(items, on="item_nbr")
        .join(transactions, on=["store_nbr", "date"])
        .with_columns(pl.col("transactions").fill_null(strategy="forward"))
        .join(national_holidays, on="date", how="left")
        .join(regional_holidays, on=["date", "state"], how="left")
        .join(local_holidays, on=["date", "city"], how="left")
        .with_columns(
            [
                pl.col("national_hol").fill_null(""),
                pl.col("regional_hol").fill_null(""),
                pl.col("local_hol").fill_null(""),
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.day().alias("day_of_month"),
                pl.col("date").dt.weekday().alias("day_of_week"),
            ]
        )
        .filter(pl.col("oil_price").is_not_null())
        .sort("traj_id", "date")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
    )
    logging.debug(f"Post join {df}")
    df.write_parquet(f"{cache_dir}/joined_df.parquet")
    gc.collect()
    return df


def read_temporal(data_dir: str) -> pl.LazyFrame:
    temporal: pl.LazyFrame = (
        pl.scan_parquet(f"{data_dir}/train.parquet")
        .drop("id")
        # cutoff dataset to reduce memory requirements.
        .filter(pl.col("date") >= _START_DATE)
        .filter(pl.col("date") <= _END_DATE)
        .with_columns([pl.col("onpromotion").map(lambda x: None if x is None else x == "True")])
        .with_columns([pl.format("{}_{}", "store_nbr", "item_nbr").alias("traj_id")])
        # remove_returns_data
        .filter(pl.col("unit_sales").min().over("traj_id") >= 0)
        .with_columns(open=pl.lit(1).cast(pl.Int8))
        .pipe(downcast_dataframe)
        .sort("date", "traj_id")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
        .upsample("date", every="1d", by="traj_id")
        .lazy()
        .with_columns(
            [
                pl.col(i).fill_null(strategy="forward")
                for i in ["store_nbr", "item_nbr", "onpromotion"]
            ]
        )
        .with_columns(pl.col("open").fill_null(0))
        .with_columns(pl.col("unit_sales").log())
        .rename({"unit_sales": "log_sales"})
        .with_columns(pl.col("log_sales").fill_null(strategy="forward"))
        .pipe(downcast_dataframe, streaming=True)
    )

    return temporal


def downcast_dataframe(df: pl.DataFrame | pl.LazyFrame, streaming: bool = False) -> pl.LazyFrame:
    columns = df.columns

    df = df.with_columns(
        [pl.col(i).cast(_COLUMN_TO_DTYPE[i]) for i in columns if i in _COLUMN_TO_DTYPE]
    )

    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=streaming)

    df = df.shrink_to_fit(in_place=True).rechunk()
    return df.lazy()


def compute_split_spec(config: Config) -> tuple[datetime, datetime]:
    validation_boundary: datetime = config.validation_boundary
    forecast_horizon = config.total_time_steps - config.encoder_steps
    test_boundary = validation_boundary + timedelta(days=forecast_horizon)
    return validation_boundary, test_boundary


def restore_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    days = df["day_of_month"]
    month = df["month"]
    year = df["year"]

    ts = [datetime(year=y, month=m, day=d).date() for d, m, y in zip(days, month, year)]
    return df.with_columns(pl.lit(ts).alias("date"))


def maybe_reload_preprocessor(data_dir: str) -> FavoritaPreprocessor | None:
    filenames = [
        f"{data_dir}/preprocessor.categorical.joblib",
        f"{data_dir}/preprocessor.real.joblib",
        f"{data_dir}/preprocessor.target.joblib",
    ]

    if all([pathlib.Path(i).is_file() for i in filenames]):
        return FavoritaPreprocessor.load(data_dir)
