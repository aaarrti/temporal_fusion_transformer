from __future__ import annotations

import glob
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import final

import polars as pl
from absl import logging
from tqdm.auto import tqdm

_NUM_IDS = 143658


@final
class Favorita:
    """
    References
    ----------

    - https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
    """

    target_feature_names = ("log_sales",)
    total_time_steps = 120
    num_encoder_steps = 90
    id_column = "traj_id"

    features = property(
        lambda self: OrderedDict(
            [
                # static
                ("item_nbr", self.integer_categorical()),
                ("store_nbr", self.integer_categorical()),
                ("city", self.string_categorical()),
                ("state", self.string_categorical()),
                ("type", self.string_categorical()),
                ("cluster", self.integer_categorical()),
                ("family", self.string_categorical()),
                ("class", self.integer_categorical()),
                ("perishable", self.integer_categorical()),
                # known real
                # ("date", FeatureSpace.float_normalized()),
                # known categorical
                ("month", self.integer_categorical()),
                ("day_of_month", self.integer_categorical()),
                ("day_of_week", self.integer_categorical()),
                ("national_hol", self.string_categorical()),
                ("regional_hol", self.string_categorical()),
                ("local_hol", self.string_categorical()),
                ("onpromotion", self.integer_categorical()),
                ("open", self.integer_categorical()),
                # observed
                ("transactions", FeatureSpace.float_normalized()),
                ("oil_price", FeatureSpace.float_normalized()),
            ]
        )
    )
    _required_files = (
        "stores.parquet",
        "items.parquet",
        "transactions.parquet",
        "oil.parquet",
        "holidays_events.parquet",
    )

    def __init__(
        self,
        start_date: datetime | None = datetime(2016, 1, 1),
        end_date: datetime | None = datetime(2016, 6, 1),
        validation_boundary=datetime(2016, 4, 1),
    ):
        """
        We assume 3 month lookback and one-month forecast horizon.

        Parameters
        ----------
        start_date:
            Limit time-series size, so don't need 40GB of RAM.
        end_date:
             Limit time-series size, so don't need 40GB of RAM.
        validation_boundary:
            Date at which validation starts.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.validation_boundary = validation_boundary

    def needs_download(self, path: str) -> bool:
        files = glob.glob(f"{path}/*.parquet")

        filenames = {i.rpartition("/")[-1] for i in files}
        missing_files = list(set(self._required_files).difference(filenames))

        if len(missing_files) == 0:
            logging.info(f"Found {files} locally, will skip download from Kaggle.")
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

    def convert_to_parquet(self, path: str):
        files = glob.glob(f"{path}/*.csv")
        for f in tqdm(files, desc="Converting to parquet"):
            f: str
            pl.scan_csv(f, try_parse_dates=True).sink_parquet(f.replace("csv", "parquet"))
            os.remove(f)

    def read_parquet(self, path: str) -> pl.DataFrame:
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
            pl.scan_parquet(f"{path}/train.parquet")
            .drop("id")
            .pipe(convert_onpromotion_to_bool)
            .pipe(downcast_dataframe)
            .pipe(filter_dates, self.start_date, self.end_date)
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

        store_info = pl.scan_parquet(f"{path}/stores.parquet").pipe(downcast_dataframe)
        items = pl.scan_parquet(f"{path}/items.parquet").pipe(downcast_dataframe)
        transactions = pl.scan_parquet(f"{path}/transactions.parquet").pipe(downcast_dataframe)
        oil = (
            pl.scan_parquet(f"{path}/oil.parquet")
            .pipe(downcast_dataframe)
            .pipe(lambda lf: lf.rename({"dcoilwtico": "oil_price"}))
        )
        holidays = pl.scan_parquet(f"{path}/holidays_events.parquet").pipe(downcast_dataframe)

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
        logging.debug(f"{df = }")
        df.write_parquet(f"{path}/df.parquet")
        return df

    def split_data(self, df: pl.DataFrame) -> [pl.DataFrame]:
        lf = df.lazy()
        forecast_horizon = self.total_time_steps - self.num_encoder_steps

        test_boundary = self.validation_boundary + timedelta(hours=forecast_horizon)

        training_df: pl.DataFrame = lf.filter(pl.col("date").over("traj_id").lt(self.validation_boundary)).collect()
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
