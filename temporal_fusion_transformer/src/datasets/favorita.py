from __future__ import annotations

import glob
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import final

import polars as pl
from absl import logging
from keras.utils import FeatureSpace
from keras.utils.data_utils import _extract_archive
from toolz.dicttoolz import valmap
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, Triple
from temporal_fusion_transformer.src.datasets.base import downcast_dataframe


_NUM_IDS = 143658


@final
class Favorita(MultiHorizonTimeSeriesDataset):
    """
    References
    ----------

    - https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
    """

    target_feature_names = ["log_sales"]
    total_time_steps = 120
    num_encoder_steps = 90

    features = property(
        lambda self: OrderedDict(
            [
                ("log_sales", FeatureSpace.float_normalized()),
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
                ("date", FeatureSpace.float_normalized()),
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
    _required_files = (
        "stores.csv",
        "items.csv",
        "transactions.csv",
        "oil.csv",
        "holidays_events.csv",
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

    def read_csv(self, path: str) -> pl.DataFrame:
        lazy_temporal = downcast_dataframe(pl.scan_csv(f"{path}/train.csv", try_parse_dates=True))

        # Filter dates to reduce storage space requirements
        if self.start_date is not None:
            lazy_temporal = lazy_temporal.filter(pl.col("date") >= self.start_date)
        if self.end_date is not None:
            lazy_temporal = lazy_temporal.filter(pl.col("date") <= self.end_date)

        logging.debug("Adding trajectory identifier")
        lazy_temporal = lazy_temporal.with_columns(
            [
                pl.format("{}_{}", "store_nbr", "item_nbr").alias("traj_id"),
            ]
        )
        # Remove all IDs with negative returns
        logging.debug("Removing returns data")

        lazy_temporal = lazy_temporal.filter(pl.col("unit_sales").min().over("traj_id") >= 0)
        lazy_temporal = lazy_temporal.with_columns(open=pl.lit(1))

        # LazyFrame does not support up-sampling
        logging.info("Up-sampling to uniform grid.")
        lazy_temporal = (
            lazy_temporal.sort("date", "traj_id")
            .collect()
            .upsample("date", every="1h", by="traj_id")
            .fill_null(strategy="forward")
            .lazy()
        )
        logging.debug(f"{lazy_temporal = }")

        store_info = downcast_dataframe(pl.scan_csv(f"{path}/stores.csv"))
        items = downcast_dataframe(pl.scan_csv(f"{path}/items.csv"))
        transactions = downcast_dataframe(pl.scan_csv(f"{path}/transactions.csv", try_parse_dates=True))

        oil = downcast_dataframe(
            pl.scan_csv(f"{path}/oil.csv", try_parse_dates=True)
            .with_columns(pl.col("dcoilwtico").forward_fill())
            .filter(pl.col("dcoilwtico").is_not_null())
        )

        holidays = downcast_dataframe(pl.scan_csv(f"{path}/holidays_events.csv", try_parse_dates=True))

        national_holidays = (
            holidays.filter(pl.col("locale") == "National")
            .select(["description", "date"])
            .rename({"description": "national_hol"})
        )
        regional_holidays = (
            holidays.filter(pl.col("locale") == "Regional")
            .select(["description", "locale_name", "date"])
            .rename({"locale_name": "state", "description": "regional_hol"})
        )
        local_holidays = (
            holidays.filter(pl.col("locale") == "Local")
            .select(["description", "locale_name", "date"])
            .rename({"locale_name": "city", "description": "local_hol"})
        )

        logging.debug("Joining tables")

        temporal: pl.LazyFrame = (
            lazy_temporal.join(oil, on="date", how="left")
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
                ]
            )
            .sort("traj_id", "date")
        )
        df = temporal.collect()
        logging.debug(f"{df = }")
        return df

    def split_data(self, df: pl.LazyFrame) -> Triple[pl.LazyFrame]:
        if not isinstance(df, pl.LazyFrame):
            df = df.lazy()

        forecast_horizon = self.total_time_steps - self.num_encoder_steps

        test_boundary = self.validation_boundary + timedelta(hours=forecast_horizon)

        training_df: pl.DataFrame = df.filter(pl.col("date").over("traj_id").lt(self.validation_boundary)).collect()
        validation_df = df.filter(pl.col("date").over("traj_id").ge(self.validation_boundary).lt(test_boundary))
        test_df = df.filter(pl.col("date").over("traj_id").ge(test_boundary))

        # Filter out identifiers not present in training (i.e. cold-started items).
        identifiers = training_df["traj_id"].unique().to_list()
        ids = set(identifiers)

        def filter_ids(frame: pl.DataFrame) -> pl.DataFrame:
            return frame.filter(pl.col("traj_id") in ids)

        validation_df = filter_ids(validation_df["valid"])
        test_df = filter_ids(test_df["test"])

        return training_df, validation_df, test_df

    def needs_download(self, path: str) -> bool:
        csvs = glob.glob(f"{path}/*.csv")

        filenames = {i.rpartition("/")[-1] for i in csvs}
        missing_files = list(set(self._required_files).difference(filenames))

        if len(missing_files) == 0:
            logging.info(f"Found {csvs} locally, will skip download from Kaggle.")
            return False
        else:
            logging.info(f"{missing_files = }")
            return True

    def download_data(self, path: str):
        # For favorita we download data from kaggle, importing kaggle at top, will cause error if no config file is found,
        # even if never invoked.
        import kaggle
        import py7zr

        kaggle.api.competition_download_files("favorita-grocery-sales-forecasting", path, quiet=False)
        logging.info("Extracting archive")
        if _extract_archive(f"{path}/favorita-grocery-sales-forecasting.zip", path):
            os.remove(f"{path}/favorita-grocery-sales-forecasting.zip")

        for i in tqdm(glob.glob(f"{path}/*.7z"), desc="Extracting 7z archives."):
            with py7zr.SevenZipFile(i, mode="r") as archive:
                archive.extractall(path=path)
            os.remove(i)
