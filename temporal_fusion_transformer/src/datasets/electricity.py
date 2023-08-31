from __future__ import annotations

import os
import pathlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from absl import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm.auto import tqdm
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from temporal_fusion_transformer.src.datasets.preprocessing import PreprocessorDict

"""
 References:
     - https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
 """

targets = ["power_usage"]
real_inputs = ["year"]
categorical_inputs = ["month", "day", "hour", "day_of_week"]
total_time_steps = 192


def convert_to_parquet(data_dir: str):
    if Path(f"{data_dir}/LD2011_2014.parquet").exists():
        logging.info(f"Found {data_dir}/LD2011_2014.parquet locally, will skip download.")
        return
    pathlib.Path(data_dir).mkdir(exist_ok=True)
    with open(f"{data_dir}/LD2011_2014.txt", "r") as file:
        txt_content = file.read()
    
    csv_content = txt_content.replace(",", ".").replace(";", ",")
    
    with open(f"{data_dir}/LD2011_2014.csv", "w+") as file:
        file.write(csv_content)
    
    pl.scan_csv(f"{data_dir}/LD2011_2014.csv", infer_schema_length=999999, try_parse_dates=True).rename(
        {"": "timestamp"}
    ).sink_parquet(f"{data_dir}/LD2011_2014.parquet")
    
    os.remove(f"{data_dir}/LD2011_2014.txt")
    os.remove(f"{data_dir}/LD2011_2014.csv")


def read_parquet(
        data_dir: str, cutoff_days: Tuple[datetime, datetime] = (datetime(2014, 1, 1), datetime(2014, 9, 8))
) -> pl.DataFrame:
    lf = pl.scan_parquet(f"{data_dir}/LD2011_2014.parquet")
    
    num_cols = lf.columns[1:]
    lf = lf.sort("timestamp")
    # down sample to 1h https://pola-rs.github.io/polars-book/user-guide/transformations/time-series/rolling/
    lf = lf.groupby_dynamic("timestamp", every="1h").agg([pl.col(i).mean() for i in num_cols])
    
    df_list = []
    
    for label in tqdm(num_cols, desc="Formatting inputs"):
        sub_df: pl.DataFrame = lf.select("timestamp", label)
        lazy_sub_df = sub_df.lazy()
        lazy_sub_df = (
            lazy_sub_df.filter(pl.col("timestamp") >= cutoff_days[0])
            .filter(pl.col("timestamp") <= cutoff_days[1])
            .rename({label: "power_usage"})
            .with_columns(
                [
                    pl.col("power_usage").cast(pl.Float32),
                    pl.col("timestamp").dt.year().alias("year").cast(pl.Float32),
                    pl.col("timestamp").dt.month().alias("month").cast(pl.UInt8),
                    pl.col("timestamp").dt.hour().alias("hour").cast(pl.UInt8),
                    pl.col("timestamp").dt.day().alias("day").cast(pl.UInt8),
                    pl.col("timestamp").dt.weekday().alias("day_of_week").cast(pl.UInt8),
                ],
                id=pl.lit(label),
            )
        )
        sub_df = lazy_sub_df.collect()
        sub_df = sub_df.shrink_to_fit(in_place=True).rechunk()
        
        df_list.append(sub_df)
    
    df: pl.DataFrame = pl.concat(df_list)
    df = df.shrink_to_fit(in_place=True).rechunk()
    logging.info(f"{df.null_count() = }")
    return df


def split_data(
        df: pl.DataFrame,
        validation_boundary: datetime = datetime(2014, 8, 8),
        test_boundary: int = datetime(2014, 9, 1),
        split_overlap_days: int = 7,
):
    """
    This dataset was recorded in interval [2011-01-01, 2015-01-01].
    """
    
    train_df = df.filter(pl.col("timestamp") < validation_boundary)
    validation_df: pl.DataFrame = (
        df.lazy()
        .filter(pl.col("timestamp") >= validation_boundary - timedelta(days=split_overlap_days))
        .filter(pl.col("timestamp") < test_boundary)
        .collect()
    )
    test_df = df.filter(pl.col("timestamp") >= test_boundary - timedelta(days=split_overlap_days))
    return (
        train_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        validation_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        test_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk()
    )


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    target_scalers = defaultdict(lambda: StandardScaler())
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())
    
    for i, sub_df in tqdm(df.groupby("id"), desc="Training scalers", total=370):
        target_scalers[i].fit(df[targets].to_numpy(order="c"))
        real_scalers[i].fit(df[real_inputs].to_numpy(order="c"))
    
    for i in tqdm(categorical_inputs, desc="Fitting label encoders"):
        label_encoders[i].fit(df[i].to_numpy())
    
    return {"real": dict(**real_scalers), "target": dict(**target_scalers), "categorical": dict(**label_encoders)}


def apply_preprocessor(
        df: pl.DataFrame,
        preprocessor: PreprocessorDict,

) -> pl.DataFrame:
    lf_list = []
    
    for i, sub_df in tqdm(df.groupby("id"), total=370, desc="Applying scalers..."):
        sub_df: pl.DataFrame
        sub_lf: pl.LazyFrame = sub_df.lazy()
        
        x_real = df[real_inputs].to_numpy(order="c")
        x_target = df[targets].to_numpy(order="c")
        
        x_real = preprocessor["real"][i].transform(x_real)
        x_target = preprocessor["target"][i].transform(x_target)
        
        sub_lf = sub_lf.with_columns(
            [pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_real, real_inputs)]
        ).with_columns(pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_target, targets))
        lf_list.append(sub_lf)
    
    df: pl.DataFrame = pl.concat(lf_list).collect()
    
    for i in tqdm(categorical_inputs):
        x = df[i].to_numpy()
        x = preprocessor["categorical"][i].transform(x)
        df = df.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Int8))
    
    df = df.shrink_to_fit(in_place=True).rechunk()
    return df


def time_series_from_array(df: pl.DataFrame) -> tf.data.Dataset:
    x = df[real_inputs + categorical_inputs + ["id"] + targets].to_numpy(order="c")
    
    num_inputs = len(real_inputs + categorical_inputs)
    
    time_series: tf.data.Dataset = timeseries_dataset_from_array(
        x,
        None,
        total_time_steps,
        batch_size=None,
    )
    time_series = time_series.map(
        lambda i: (i[..., :num_inputs], i[..., num_inputs:]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    
    return time_series


def make_dataset(
        data_dir: str,
        validation_boundary: datetime = datetime(2014, 8, 8),
        test_boundary: int = datetime(2014, 9, 1),
        split_overlap_days: int = 7,
        cutoff_days: Tuple[datetime, datetime] = (datetime(2014, 1, 1), datetime(2014, 9, 8)),
) -> Tuple[
    Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset,
    ], PreprocessorDict]:
    convert_to_parquet(data_dir)
    df = read_parquet(data_dir, cutoff_days=cutoff_days)
    logging.info(f"{df.columns = }")
    preprocessor = train_preprocessor(df)
    
    training_df, validation_df, test_df = split_data(df, validation_boundary, test_boundary, split_overlap_days)
    training_df = apply_preprocessor(training_df, preprocessor)
    validation_df = apply_preprocessor(validation_df, preprocessor)
    test_df = apply_preprocessor(test_df, preprocessor)
    # convert to time-series
    training_time_series = time_series_from_array(training_df)
    validation_time_series = time_series_from_array(validation_df)
    test_time_series = time_series_from_array(test_df)
    return (training_time_series, validation_time_series, test_time_series), preprocessor
