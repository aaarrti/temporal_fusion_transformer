from __future__ import annotations

import glob
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Final, Tuple, List

import polars as pl
from absl import logging
from tqdm.auto import tqdm

import tensorflow as tf

from temporal_fusion_transformer.src.datasets.preprocessing import PreprocessorDict

num_ids: Final[int] = 143658
total_time_steps: Final[int] = 120
num_encoder_steps: Final[int] = 90
required_files: Final[List[str]] = [
    "stores.parquet",
    "items.parquet",
    "transactions.parquet",
    "oil.parquet",
    "holidays_events.parquet",
]


def convert_to_parquet(data_dir: str):
    files = glob.glob(f"{data_dir}/*.parquet")

    filenames = {i.rpartition("/")[-1] for i in files}
    missing_files = list(set(required_files).difference(filenames))

    if len(missing_files) == 0:
        logging.info(f"Found {files}.")
        return

    files = glob.glob(f"{data_dir}/*.csv")
    for f in tqdm(files, desc="Converting to parquet"):
        f: str
        pl.scan_csv(f, try_parse_dates=True).sink_parquet(f.replace("csv", "parquet"))
        os.remove(f)


def read_parquet(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
) -> pl.DataFrame:
    def downcast_dataframe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            [
                pl.col("onpromotion").cast(pl.Int8),
                pl.col("store_nbr").cast(pl.Int8),
                pl.col("item_nbr").cast(pl.Int32),
                pl.col("unit_sales").cast(pl.Float32),
            ]
        )

    def convert_onpromotion_to_bool(lf: pl.LazyFrame) -> pl.LazyFrame:
        def map_fn(x):
            if x is None:
                return None
            else:
                return x == "True"

        return lf.with_columns([pl.col("onpromotion").apply(map_fn)])

    def remove_returns_data(lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = lf.filter(pl.col("unit_sales").min().over("traj_id") >= 0)
        lf = lf.with_columns(open=pl.lit(1).cast(pl.Int8))
        return lf

    def filter_dates(lf: pl.LazyFrame, start_date, end_date) -> pl.LazyFrame:
        # Filter dates to reduce storage space requirements
        if start_date is not None:
            lf = lf.filter(pl.col("date") >= start_date)
        if end_date is not None:
            lf = lf.filter(pl.col("date") <= end_date)
        return lf

    temporal: pl.LazyFrame = (
        pl.scan_parquet(f"{data_dir}/train.parquet")
        .drop("id")
        .pipe(convert_onpromotion_to_bool)
        .pipe(downcast_dataframe)
        .pipe(filter_dates, start_date, end_date)
        .with_columns([pl.format("{}_{}", "store_nbr", "item_nbr").alias("traj_id")])
        .pipe(remove_returns_data)
        .sort("date", "traj_id")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
        .upsample("date", every="1d", by="traj_id")
        .fill_null(strategy="forward")
        .with_columns(pl.col("unit_sales").log())
        .rename({"unit_sales": "log_sales"})
        .lazy()
    )

    store_info = pl.scan_parquet(f"{data_dir}/stores.parquet").pipe(downcast_dataframe)
    items = pl.scan_parquet(f"{data_dir}/items.parquet").pipe(downcast_dataframe)
    transactions = pl.scan_parquet(f"{data_dir}/transactions.parquet").pipe(downcast_dataframe)
    oil = (
        pl.scan_parquet(f"{data_dir}/oil.parquet")
        .pipe(downcast_dataframe)
        .pipe(lambda lf: lf.rename({"dcoilwtico": "oil_price"}))
    )
    holidays = pl.scan_parquet(f"{data_dir}/holidays_events.parquet").pipe(downcast_dataframe)

    national_holidays = (
        holidays.filter(pl.col("locale") == "National")
        .select(["description", "date"])
        .rename({"description": "national_hol"})
        .collect()
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )
    regional_holidays = (
        holidays.filter(pl.col("locale") == "Regional")
        .select(["description", "locale_name", "date"])
        .rename({"locale_name": "state", "description": "regional_hol"})
        .collect()
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )
    local_holidays = (
        holidays.filter(pl.col("locale") == "Local")
        .select(["description", "locale_name", "date"])
        .rename({"locale_name": "city", "description": "local_hol"})
        .collect()
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )

    logging.debug("Joining tables")

    df: pl.DataFrame = (
        temporal.join(oil, on="date", how="left")
        .join(store_info, on="store_nbr")
        .join(items, on="item_nbr")
        .join(transactions, on=["store_nbr", "date"])
        .join(national_holidays, on="date", how="left")
        .join(regional_holidays, on=["date", "state"], how="left")
        .join(local_holidays, on=["date", "city"], how="left")
        .with_columns(
            [
                pl.col("oil_price").fill_null(strategy="forward"),
                pl.col("national_hol").fill_null(""),
                pl.col("regional_hol").fill_null(""),
                pl.col("local_hol").fill_null(""),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.day().alias("day_of_month"),
                pl.col("date").dt.weekday().alias("day_of_week"),
            ]
        )
        .filter(pl.col("oil_price").is_not_null())
        .sort("date", "traj_id")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
    )
    logging.debug(f"{df.null_count() = }")
    # df.write_parquet(f"{data_dir}/df.parquet")
    return df


def split_data(
    df: pl.DataFrame,
    validation_boundary=datetime(2016, 4, 1),
) -> [pl.DataFrame]:
    lf = df.lazy()
    forecast_horizon = total_time_steps - num_encoder_steps

    test_boundary = validation_boundary + timedelta(hours=forecast_horizon)

    training_df: pl.DataFrame = lf.filter(pl.col("date").over("traj_id").lt(validation_boundary)).collect()
    validation_df = df.filter(pl.col("date").over("traj_id").ge(validation_boundary).lt(test_boundary))
    test_df = df.filter(pl.col("date").over("traj_id").ge(test_boundary))

    # Filter out identifiers not present in training (i.e. cold-started items).
    identifiers = training_df["traj_id"].unique().to_list()
    ids = set(identifiers)

    def filter_ids(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("traj_id") in ids)

    validation_df = filter_ids(validation_df["valid"])
    test_df = filter_ids(test_df["test"])

    return training_df, validation_df, test_df


def make_dataset(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], PreprocessorDict]:
    convert_to_parquet(data_dir)
    df = read_parquet(data_dir, start_date, end_date)
