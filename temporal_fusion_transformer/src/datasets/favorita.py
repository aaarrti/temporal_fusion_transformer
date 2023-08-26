from __future__ import annotations

import glob
import os
from collections import OrderedDict
from datetime import datetime
from typing import final

import polars as pl
from absl import logging
from keras.utils.data_utils import _extract_archive
from keras.utils import FeatureSpace
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, Triple

_DEFAULT_START_DATE = datetime(2015, 1, 1)
_DEFAULT_END_DATE = datetime(2016, 6, 1)


@final
class Favorita(MultiHorizonTimeSeriesDataset):
    target_feature_names = ["log_sales"]
    total_time_steps = 120

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
        "holiday_events.csv",
    )

    def __init__(
        self,
        start_date: datetime | None = _DEFAULT_START_DATE,
        end_date: datetime | None = _DEFAULT_END_DATE,
        validation_boundary=datetime(2015, 12, 1),
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.validation_boundary = validation_boundary

    def read_csv(self, path: str) -> pl.LazyFrame:
        return read_raw_csv(path, self.start_date, self.end_date)

    def split_data(self, df: pl.LazyFrame) -> Triple[pl.LazyFrame]:
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
    
    def needs_download(self, path: str) -> bool:
        csvs = glob.glob(f"{path}/*.csv")
        
        filenames = {i.rpartition("/")[-1] for i in csvs}
        missing_files = list(set(self._required_files).difference(filenames))
        
        if len(missing_files) == 0:
            logging.info(f"Found {csvs} locally, will skip download from Kaggle.")
            return False
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


def read_raw_csv(
    data_folder: str,
    start_date: datetime | None = _DEFAULT_START_DATE,
    end_date: datetime | None = _DEFAULT_END_DATE,
) -> pl.LazyFrame:
    temporal = make_temporal_dataframe(data_folder, start_date, end_date)
    store_info = pl.read_csv(f"{data_folder}/stores.csv")
    items = pl.read_csv(f"{data_folder}/items.csv")
    transactions = pl.read_csv(f"{data_folder}/transactions.csv", try_parse_dates=True)

    oil = (
        pl.read_csv(f"{data_folder}/oil.csv", try_parse_dates=True)
        .with_columns(pl.col("dcoilwtico").forward_fill())
        .filter(pl.col("dcoilwtico").is_not_null())
    )

    holidays = pl.read_csv(f"{data_folder}/holidays_events.csv", try_parse_dates=True)

    national_holidays = holidays.filter(pl.col("locale") == "National")
    regional_holidays = holidays.filter(pl.col("locale") == "Regional").rename({"locale_name": "state"})
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
    start_date: datetime | None = _DEFAULT_START_DATE,
    end_date: datetime | None = _DEFAULT_END_DATE,
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
            pl.format("{}_{}_{}", "store_nbr", "item_nbr", "da"
                                                           "te").alias("unique_id"),
        ]
    )
    # Remove all IDs with negative returns
    logging.info("Removing returns data")

    lazy_df = lazy_df.filter(pl.col("unit_sales").min().over("traj_id") >= 0)
    lazy_df = lazy_df.with_columns(open=pl.lit(1))

    df = lazy_df.collect()
    columns_to_resample = [i for i in lazy_df.columns if i not in ("traj_id", "date")]
    logging.debug(f"{columns_to_resample = }")

    # Resampling
    resampled_dfs = []
    pbar = tqdm(list(df.groupby("traj_id")), desc="Resampling")
    for traj_id, sub_df in pbar:
        sub_df: pl.DataFrame
        lazy_sub_df = (
            sub_df.lazy()
            .sort(by="date")
            .groupby_dynamic("date", every="1d")
            .agg([pl.col(i).mean() for i in columns_to_resample])
            .with_columns(
                [
                    #pl.col("store_nbr").forward_fill(),
                    #pl.col("item_nbr").forward_fill(),
                    #pl.col("onpromotion").forward_fill(),
                    #pl.col("open").fill_null(0),
                    pl.col("unit_sales").log().alias("log_sales"),
                ]
            )
            .drop(["unit_sales"])
        )

        resampled_dfs.append(lazy_sub_df)

    lazy_df = pl.concat(resampled_dfs).collect()
    return lazy_df
