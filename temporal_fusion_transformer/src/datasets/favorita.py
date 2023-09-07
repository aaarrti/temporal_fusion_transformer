from __future__ import annotations

import glob
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Mapping, Tuple, TypedDict

import polars as pl
import tensorflow as tf
from absl import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.datasets.preprocessing import (
    time_series_from_array,
)


class PreprocessorDict(TypedDict):
    real: Mapping[str, StandardScaler]
    target: StandardScaler
    categorical: Mapping[str, LabelEncoder]


NUM_IDS = 143658

REQUIRED_FILES = [
    "stores.parquet",
    "items.parquet",
    "transactions.parquet",
    "oil.parquet",
    "holidays_events.parquet",
]
REAL_INPUTS = [
    # observed
    "oil_price",
    "transactions",
]
CATEGORICAL_INPUTS = [
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
    "day of month",
    "day of week",
    "national holiday" "regional hol",
    "local holiday",
    "onpromotion",
    "open",
]


def convert_to_parquet(data_dir: str):
    files = glob.glob(f"{data_dir}/*.parquet")

    filenames = {i.rpartition("/")[-1] for i in files}
    missing_files = list(set(REQUIRED_FILES).difference(filenames))

    if len(missing_files) == 0:
        logging.info(f"Found {files} locally.")
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
    def remove_returns_data(lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = lf.filter(pl.col("unit_sales").min().over("traj_id") >= 0)
        lf = lf.with_columns(open=pl.lit(1).cast(pl.Int8))
        return lf

    def filter_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Filter dates to reduce storage space requirements
        if start_date is not None:
            lf = lf.filter(pl.col("date") >= start_date)
        if end_date is not None:
            lf = lf.filter(pl.col("date") <= end_date)
        return lf

    temporal: pl.LazyFrame = (
        pl.scan_parquet(f"{data_dir}/train.parquet")
        .drop("id")
        .pipe(filter_dates)
        .with_columns([pl.col("onpromotion").map(lambda x: None if x is None else x == "True")])
        .with_columns(
            [
                pl.col("onpromotion").cast(pl.UInt8),
                pl.col("store_nbr").cast(pl.UInt8),
                pl.col("item_nbr").cast(pl.UInt32),
                pl.col("unit_sales").cast(pl.Float32),
            ]
        )
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

    store_info = (
        pl.read_parquet(f"{data_dir}/stores.parquet")
        .with_columns(pl.col("cluster").cast(pl.UInt8))
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )

    items = (
        pl.read_parquet(f"{data_dir}/items.parquet")
        .with_columns([pl.col("perishable").cast(pl.UInt8), pl.col("class").cast(pl.UInt16)])
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )
    transactions = (
        pl.scan_parquet(f"{data_dir}/transactions.parquet")
        .with_columns([pl.col("store_nbr").cast(pl.UInt8), pl.col("transactions").cast(pl.UInt16)])
        .collect()
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )
    oil = (
        pl.read_parquet(f"{data_dir}/oil.parquet")
        .rename({"dcoilwtico": "oil_price"})
        .with_columns(pl.col("oil_price").cast(pl.Float32))
        .shrink_to_fit(in_place=True)
        .rechunk()
        .lazy()
    )
    holidays = pl.scan_parquet(f"{data_dir}/holidays_events.parquet")

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
    df.write_parquet(f"{data_dir}/df.parquet")
    return df


def split_data(
    df: pl.DataFrame, validation_boundary=datetime(2016, 4, 1), total_time_steps: int = 120, num_encoder_steps: int = 90
) -> [pl.DataFrame]:
    lf = df.lazy()
    forecast_horizon = total_time_steps - num_encoder_steps

    test_boundary = validation_boundary + timedelta(hours=forecast_horizon)

    training_df: pl.DataFrame = lf.filter(pl.col("date").over("traj_id").lt(validation_boundary)).collect()
    validation_df = df.filter(pl.col("date").over("traj_id").ge(validation_boundary)).filter(
        pl.col("date").over("traj_id").lt(test_boundary)
    )
    test_df = df.filter(pl.col("date").over("traj_id").ge(test_boundary))

    # Filter out identifiers not present in training (i.e. cold-started items).
    identifiers = training_df["traj_id"].unique().to_list()
    ids = set(identifiers)

    def filter_ids(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("traj_id") in ids)

    validation_df = filter_ids(validation_df["valid"])
    test_df = filter_ids(test_df["test"])

    return training_df.drop("traj_id"), validation_df.drop("traj_id"), test_df.drop("traj_id")


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    # In contrast to electricity for favorita we don't group before training StandardScaler
    target_scaler = StandardScaler()
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())
    # label_encoders["traj_id"].fit(df["traj_id"].to_numpy())
    target_scaler.fit(df["log_sales"].to_numpy().reshape(1, -1))
    for i in REAL_INPUTS:
        real_scalers[i].fit(df[i].to_numpy().reshape(1, -1))

    for i in tqdm(CATEGORICAL_INPUTS, desc="Fitting label encoders"):
        label_encoders[i].fit(df[i].to_numpy())

    return {"real": dict(**real_scalers), "target": target_scaler, "categorical": dict(**label_encoders)}


def apply_preprocessor(
    df: pl.DataFrame,
    preprocessor: PreprocessorDict,
) -> pl.DataFrame:
    lf = df.lazy()

    log_sales = preprocessor["target"].transform(df["log_sales"].to_numpy())
    lf = lf.drop("log_sales").with_columns(log_sales=pl.lit(log_sales).cast(pl.Float32))

    for i in tqdm(REAL_INPUTS):
        x = df[i].to_numpy()
        x = preprocessor["real"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Float32))

    for i in tqdm(CATEGORICAL_INPUTS):
        x = df[i].to_numpy()
        x = preprocessor["categorical"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Int8))

    df = lf.collect().shrink_to_fit(in_place=True).rechunk()
    return df


def make_dataset(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
    validation_boundary=datetime(2016, 4, 1),
    total_time_steps: int = 120,
    num_encoder_steps: int = 90,
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], PreprocessorDict]:
    convert_to_parquet(data_dir)
    df = read_parquet(data_dir, start_date, end_date)
    logging.info(f"{df.columns = }")
    preprocessor = train_preprocessor(df)
    training_df, validation_df, test_df = split_data(
        df, validation_boundary, total_time_steps=total_time_steps, num_encoder_steps=num_encoder_steps
    )
    training_df = apply_preprocessor(training_df, preprocessor)
    validation_df = apply_preprocessor(validation_df, preprocessor)
    test_df = apply_preprocessor(test_df, preprocessor)
    # ---

    inputs = REAL_INPUTS + CATEGORICAL_INPUTS

    training_time_series = time_series_from_array(
        training_df,
        inputs=inputs,
        targets=["log_sales"],
        total_time_steps=total_time_steps,
    )
    validation_time_series = time_series_from_array(
        validation_df,
        inputs=inputs,
        targets=["log_sales"],
        total_time_steps=total_time_steps,
    )
    test_time_series = time_series_from_array(
        test_df,
        inputs=inputs,
        targets=["log_sales"],
        total_time_steps=total_time_steps,
    )
    return (training_time_series, validation_time_series, test_time_series), preprocessor
