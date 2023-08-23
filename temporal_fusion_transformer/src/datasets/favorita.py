from __future__ import annotations

import glob
import os
from collections import OrderedDict
from datetime import datetime
from typing import final

import polars as pl
from absl import logging
from absl_extra import keras_pbar
from keras.utils.data_utils import _extract_archive
from keras.utils import FeatureSpace
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, Triple


@final
class Favorita(MultiHorizonTimeSeriesDataset):
    target_feature_names = ["log_sales"]
    total_time_steps = 120

    features = property(
        lambda self: OrderedDict(
            [
                ("log_sales", FeatureSpace.float()),
                # static
                ("item_nbr", self.integer_categorical()),
                ("store_nbr", self.integer_categorical()),
                ("city", self.string_categorical()),
                ("state", self.string_categorical()),
                ("type", self.string_categorical()),
                ("cluster", self.integer_categorical()),
                ("family", self.string_categorical()),
                ("class", self.integer_categorical()),  # mb real?
                ("perishable", self.integer_categorical()),
                # known real
                ("date", FeatureSpace.float()),
                # known categorical
                ("day_of_month", self.integer_categorical()),
                ("month", self.integer_categorical()),
                ("open", self.integer_categorical()),
                ("onpromotion", self.integer_categorical()),
                ("day_of_week", self.integer_categorical()),
                ("national_hol", self.string_categorical()),
                ("regional_hol", self.string_categorical()),
                ("local_hol", self.string_categorical()),
                # observed
                ("transactions", FeatureSpace.float_normalized()),
                ("oil", FeatureSpace.float_normalized()),
            ]
        )
    )

    def __init__(
        self,
        start_date: datetime | None = datetime(2015, 1, 1),
        end_date: datetime | None = datetime(2016, 6, 1),
        validation_boundary=datetime(2015, 12, 1),
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.validation_boundary = validation_boundary

    def read_csv(self, path: str) -> pl.LazyFrame:
        return read_raw_csv(path, self.start_date, self.end_date)

    def split_data(self, df: pl.LazyFrame) -> Triple[pl.LazyFrame]:
        from absl_extra.keras_pbar import keras_pbar

        time_steps = self.fixed_parameters.total_time_steps
        lookback = self.fixed_parameters.num_encoder_steps
        forecast_horizon = time_steps - lookback

        df["date"] = pd.to_datetime(df["date"])
        df_lists = {"train": [], "valid": [], "test": []}
        for _, sliced in keras_pbar(df.groupby("traj_id")):
            index = sliced["date"]
            train = sliced.loc[index < self.validation_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            valid = sliced.iloc[train_len - lookback : valid_len, :]
            test = sliced.iloc[valid_len - lookback : valid_len + forecast_horizon, :]

            sliced_map = {"train": train, "valid": valid, "test": test}

            for k in sliced_map:
                item = sliced_map[k]

                if len(item) >= time_steps:
                    df_lists[k].append(item)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

        train = dfs["train"]
        # Filter out identifiers not present in training (i.e. cold-started items).
        identifiers = list(df[self.id_column].unique())

        def filter_ids(frame):
            ids = set(identifiers)
            index = frame["traj_id"]
            return frame.loc[index.apply(lambda x: x in ids)]

        valid = filter_ids(dfs["valid"])
        test = filter_ids(dfs["test"])

        return train, valid, test

    def download_data(self, path: str):
        # For favorita we download data from kaggle, and upload it to GCS.
        # importing kaggle at top, will cause error if no config file is found, event if never invoked.
        import kaggle
        import py7zr

        csvs = glob.glob(f"{path}/*.csv")

        if len(csvs) == 0:
            kaggle.api.competition_download_files("favorita-grocery-sales-forecasting", path, quiet=False)
            logging.info("Extracting archive")
            if _extract_archive(f"{path}/favorita-grocery-sales-forecasting.zip", path):
                os.remove(f"{path}/favorita-grocery-sales-forecasting.zip")

            for i in keras_pbar(glob.glob(f"{path}/*.7z")):
                with py7zr.SevenZipFile(i, mode="r") as archive:
                    archive.extractall(path=path)
                os.remove(i)
        else:
            logging.info(f"Found {csvs}, skipping download from Kaggle.")


def read_raw_csv(
    data_folder: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pl.LazyFrame:
    temporal = make_temporal_dataframe(data_folder, start_date, end_date)
    store_info = pl.scan_csv(f"{data_folder}/stores.csv")
    items = pl.scan_csv(f"{data_folder}/items.csv")
    transactions = pl.scan_csv(f"{data_folder}/transactions.csv", try_parse_dates=True)

    oil = (
        pl.scan_csv(f"{data_folder}/oil.csv", try_parse_dates=True)
        .with_columns(pl.col("dcoilwtico").forward_fill())
        .filter(pl.col("dcoilwtico").is_not_null())
    )

    holidays = pl.scan_csv(f"{data_folder}/holidays_events.csv", try_parse_dates=True)

    national_holidays = holidays.filter(pl.col("Locale") == "National")
    regional_holidays = holidays.filter(pl.col("Locale") == "Regional").rename({"locale_name": "state"})
    local_holidays = holidays.filter(pl.col("locale") == "Local").rename({"locale_name": "city"})

    temporal = (
        temporal.join(oil, on="date", how="left", suffix="oil_")
        .join(store_info, on="store_nbr", how="left", suffix="store_info")
        .join(items, on="item_nbr", how="left", suffix="items_")
        .join(transactions, on=["date", "store_nbr"], suffix="transactions_")
        .join(national_holidays, on="date", suffix="national_holidays_")
        .join(regional_holidays, on=["date", "sate"], suffix="regional_holidays_")
        .join(local_holidays, on=["date", "city"], suffix="local_holidays_")
        .sort("unique_id")
    )

    return temporal


def make_temporal_dataframe(
    data_folder: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pl.LazyFrame:
    # load temporal data
    lazy_df = pl.scan_csv(f"{data_folder}/train.csv", try_parse_dates=True)

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        lazy_df = lazy_df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        lazy_df = lazy_df.filter(pl.col("date") <= end_date)

    logging.info("Adding trajectory identifier")
    # Add trajectory identifier

    lazy_df = lazy_df.with_columns(
        [
            pl.format("{}_{}", "store_nbr", "item_nbr").alias("traj_id"),
            pl.format("{}_{}_{}", "store_nbr", "item_nbr", "date").alias("unique_id"),
        ]
    )
    # Remove all IDs with negative returns
    logging.info("Removing returns data")

    lazy_df = lazy_df.filter(pl.col("unit_sales").min().over("traj_id") >= 0)
    lazy_df = lazy_df.with_columns(open=pl.lit(1))

    df = lazy_df.collect()
    columns_to_resample = [i for i in df.columns if i != "traj_id"]

    # Resampling
    resampled_dfs = []
    for traj_id, sub_df in (pbar := tqdm(df.groupby("traj_id"))):
        pbar.desc = f"Resampling to {traj_id}"
        lazy_sub_df: pl.LazyFrame = sub_df.lazy()
        lazy_sub_df = (
            lazy_sub_df.groupby_dynamic("traj_id", every="1d")
            .agg([pl.col(i).mean() for i in columns_to_resample])
            .with_columns(
                [
                    pl.col("store_nbr").forward_fill(),
                    pl.col("item_nbr").forward_fill(),
                    pl.col("onpromotion").forward_fill(),
                    pl.col("open").fill_null(0),
                    pl.col("log_sales").log().alias("log_sales"),
                ]
            )
            .drop(["unit_sales"])
        )

        resampled_dfs.append(lazy_sub_df)

    lazy_df = pl.concat(resampled_dfs)
    return lazy_df
