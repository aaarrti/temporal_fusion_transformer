from __future__ import annotations

import gc
import logging
from datetime import datetime
from importlib import util
from pathlib import Path

import polars as pl
from sklearn.utils.validation import _is_fitted  # noqa

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


class FavoritaPreprocessor:
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
