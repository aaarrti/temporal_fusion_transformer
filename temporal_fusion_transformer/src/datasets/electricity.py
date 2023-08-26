from __future__ import annotations

import os
import pathlib
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, final

import polars as pl
from absl import logging
from keras.utils import FeatureSpace, get_file
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, Triple


class CutoffDays(NamedTuple):
    start: int
    stop: int


CUTOFF_DAYS = CutoffDays(start=1096, stop=1346)
SPLIT_OVERLAP_DAYS = 7


@final
class Electricity(MultiHorizonTimeSeriesDataset):
    """
    References:
        - https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    """

    total_time_steps = 8 * 24
    target_feature_names = ("power_usage",)
    features = cached_property(
        lambda self: OrderedDict(
            [
                # target
                ("power_usage", FeatureSpace.float_normalized()),
                # static
                ("id", self.string_categorical()),
                # known real
                ("hours_from_start", FeatureSpace.float_normalized()),
                ("days_from_start", FeatureSpace.float_normalized()),
                # known categorical
                ("day", self.integer_categorical()),
                ("hour", self.integer_categorical()),
                ("day_of_week", self.integer_categorical()),
                ("month", self.integer_categorical()),
            ]
        )
    )

    def __init__(
        self,
        validation_boundary: int = 1315,
        test_boundary: int = 1339,
    ):
        """

        Parameters
        ----------
        validation_boundary:
            Starting year for validation data
        test_boundary:
            Starting year for test data
        """
        self.validation_boundary = validation_boundary
        self.test_boundary = test_boundary

    def split_data(self, df: pl.DataFrame) -> Triple[pl.DataFrame]:
        train_df = df.filter(pl.col("days_from_start") < self.validation_boundary)
        validation_df = (
            df.lazy()
            .filter(pl.col("days_from_start") >= self.validation_boundary - SPLIT_OVERLAP_DAYS)
            .filter(pl.col("days_from_start") < self.test_boundary)
            .collect()
        )
        test_df = df.filter(pl.col("days_from_start") >= self.test_boundary - SPLIT_OVERLAP_DAYS)
        return train_df, validation_df, test_df

    def needs_download(self, path: str) -> bool:
        if Path(f"{path}/LD2011_2014.csv").exists():
            logging.info(f"Found {path}/LD2011_2014.csv locally, will skip download.")
            return False
        else:
            return True

    def download_data(self, path: str):
        pathlib.Path(path).mkdir(exist_ok=True)
        logging.info(f"Downloading LD2011_2014.txt.zip")
        get_file(
            origin=f"https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
            cache_dir=path,
            extract=True,
            archive_format="zip",
            cache_subdir=".",
        )
        os.remove(f"{path}/LD2011_2014.txt.zip")

        with open(f"{path}/LD2011_2014.txt", "r") as file:
            txt_content = file.read()

        csv_content = txt_content.replace(",", ".").replace(";", ",")

        with open(f"{path}/LD2011_2014.csv", "w+") as file:
            file.write(csv_content)

    def read_csv(self, path: str) -> pl.DataFrame:
        return read_raw_csv(path)


def read_raw_csv(path: str) -> pl.DataFrame:
    lazy_df = pl.scan_csv(f"{path}/LD2011_2014.csv", infer_schema_length=999999, try_parse_dates=True)
    lazy_df = lazy_df.rename({"": "timestamp"})

    num_cols = lazy_df.columns[1:]
    lazy_df = lazy_df.sort(by="timestamp")
    # down sample to 1h https://pola-rs.github.io/polars-book/user-guide/transformations/time-series/rolling/
    lazy_df = lazy_df.groupby_dynamic("timestamp", every="1h").agg([pl.col(i).mean() for i in num_cols])
    # replace dummy 0.0 by nulls
    df: pl.DataFrame = lazy_df.with_columns(
        [pl.col(c).apply(lambda i: None if i == 0.0 else i) for c in num_cols]
    ).collect()

    earliest_time = df["timestamp"].min()

    def make_hours_from_start(date: pl.datetime):
        return (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24

    df_list = []

    for label in tqdm(num_cols, desc="Formatting inputs"):
        sub_df: pl.DataFrame = df.select("timestamp", label)

        start_date = (
            sub_df.select("timestamp", pl.col(label).forward_fill())
            .filter(pl.col(label).is_not_null())["timestamp"]
            .min()
        )
        end_date = (
            sub_df.select("timestamp", pl.col(label).backward_fill())
            .filter(pl.col(label).is_not_null())["timestamp"]
            .max()
        )

        lazy_sub_df = sub_df.lazy()

        lazy_sub_df = (
            lazy_sub_df.filter(pl.col("timestamp") >= start_date)
            .filter(pl.col("timestamp") <= end_date)
            .rename({label: "power_usage"})
            .with_columns(
                [
                    pl.col("power_usage").fill_null(value=0.0),
                    pl.col("timestamp").apply(make_hours_from_start).alias("hours_from_start"),
                    pl.col("timestamp").apply(lambda date: (date - earliest_time).days).alias("days_from_start"),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.day().alias("day"),
                    pl.col("timestamp").dt.weekday().alias("day_of_week"),
                    pl.col("timestamp").dt.month().alias("month"),
                ],
                id=pl.lit(label),
            )
            .drop("timestamp")
        )

        df_list.append(lazy_sub_df)

    df: pl.LazyFrame = pl.concat(df_list)
    del df_list

    # Filter to match range used by other academic papers
    return (
        df.filter(pl.col("days_from_start") >= CUTOFF_DAYS.start)
        .filter(pl.col("days_from_start") <= CUTOFF_DAYS.stop)
        .collect()
    )
