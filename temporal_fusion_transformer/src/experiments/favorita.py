from __future__ import annotations

import gc
import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from importlib import util
from typing import TYPE_CHECKING, Literal, Mapping, Tuple, TypedDict, Callable

import numpy as np
from absl import logging
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.experiments.base import (
    DataPreprocessorBase,
    MultiHorizonTimeSeriesDataset,
    TrainerBase,
)
from temporal_fusion_transformer.src.experiments.configs.fixed_parameters import get_config
from temporal_fusion_transformer.src.experiments.util import (
    time_series_dataset_from_dataframe,
    persist_dataset,
    count_groups,
)

if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import ConfigDict
    from temporal_fusion_transformer.src.training.metrics import MetricContainer
    from temporal_fusion_transformer.src.training.training_lib import (
        TrainStateContainer,
    )

try:
    import polars as pl
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ModuleNotFoundError as ex:
    logging.warning(ex)


class Favorita(MultiHorizonTimeSeriesDataset):
    def __init__(
        self,
        start_date: datetime | None = datetime(2016, 1, 1),
        end_date: datetime | None = datetime(2016, 6, 1),
        validation_boundary: datetime = datetime(2016, 4, 1),
    ):
        config = get_config("favorita")
        self.start_date = start_date
        self.end_date = end_date
        self.validation_boundary = validation_boundary
        self.total_time_steps = config.total_time_steps
        self.num_encoder_steps = config.num_encoder_steps

    @property
    def trainer(self) -> TrainerBase:
        return Trainer()

    def make_dataset(
        self, data_dir: str, save_dir: str | None = None
    ) -> None | Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessorBase]:
        training_ds, validation_ds, test_df, preprocessor = make_dataset(
            data_dir=data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            validation_boundary=self.validation_boundary,
            total_time_steps=self.total_time_steps,
            num_encoder_steps=self.num_encoder_steps,
        )
        if save_dir is not None:
            persist_dataset(
                training_ds, validation_ds, test_df, preprocessor.preprocessor, save_dir
            )
        else:
            return training_ds, validation_ds, test_df, preprocessor

    def convert_to_parquet(
        self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True
    ):
        convert_to_parquet(download_dir, output_dir, delete_processed=delete_processed)


class DataPreprocessor(DataPreprocessorBase):
    def __init__(self, preprocessor: PreprocessorDict):
        self.preprocessor = preprocessor

    @staticmethod
    def load(file_name: str) -> DataPreprocessorBase:
        pass

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def convert_dataframe_to_tf_dataset(self, df: pl.DataFrame) -> tf.data.Dataset:
        pass

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        pass


class Trainer(TrainerBase):
    def run(
        self,
        data_dir: str,
        batch_size: int,
        config: ConfigDict,
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        save_path: str | None = None,
        verbose: bool = True,
        profile: bool = False,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        pass

    def run_distributed(
        self,
        data_dir: str,
        batch_size: int,
        config: ConfigDict,
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        save_path: str | None = None,
        verbose: bool = True,
        profile: bool = False,
        device_type: Literal["gpu", "tpu"] = "gpu",
        prefetch_buffer_size: int = 2,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        pass


# ----------------- actual implementations --------------


class PreprocessorDict(TypedDict):
    real: Mapping[str, StandardScaler]
    target: StandardScaler
    categorical: Mapping[str, LabelEncoder]


NUM_IDS = 143658

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
]
_TARGETS = ["log_sales"]

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


def make_dataset(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
    validation_boundary: datetime = datetime(2016, 4, 1),
    total_time_steps: int = 120,
    num_encoder_steps: int = 90,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, PreprocessorDict]:
    df = read_parquet(data_dir, start_date, end_date)
    preprocessor = train_preprocessor(df)
    training_df, validation_df, test_df = split_data(
        df,
        validation_boundary,
        total_time_steps=total_time_steps,
        num_encoder_steps=num_encoder_steps,
    )

    make_dataset_fn: Callable[[pl.DataFrame], tf.data.Dataset] = partial(
        time_series_dataset_from_dataframe,
        inputs=_REAL_INPUTS + _CATEGORICAL_INPUTS,
        targets=["log_sales"],
        total_time_steps=total_time_steps,
        id_column="traj_id",
        preprocess_fn=partial(apply_preprocessor, preprocessor=preprocessor),
    )

    training_ds = make_dataset_fn(training_df)
    validation_ds = make_dataset_fn(validation_df)
    return training_ds, validation_ds, test_df


def convert_to_parquet(data_dir: str, output_dir: str | None = None, delete_processed: bool = True):
    if output_dir is None:
        output_dir = data_dir

    files = glob(f"{data_dir}/*.csv")
    for file in tqdm(files, desc="Converting to parquet"):
        file: str
        target_file = file.replace(data_dir, output_dir).replace("csv", "parquet")
        pl.scan_csv(file, try_parse_dates=True).sink_parquet(target_file)
        if delete_processed:
            os.remove(file)


def split_data(
    df: pl.DataFrame,
    validation_boundary=datetime(2016, 4, 1),
    total_time_steps: int = 120,
    num_encoder_steps: int = 90,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    lf = df.lazy()
    forecast_horizon = total_time_steps - num_encoder_steps

    test_boundary = validation_boundary + timedelta(hours=forecast_horizon)

    training_df: pl.DataFrame = lf.filter(
        pl.col("date").over("traj_id").lt(validation_boundary)
    ).collect()
    validation_df = df.filter(pl.col("date").over("traj_id").ge(validation_boundary)).filter(
        pl.col("date").over("traj_id").lt(test_boundary)
    )
    test_df = df.filter(pl.col("date").over("traj_id").ge(test_boundary))

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
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
) -> pl.DataFrame:
    temporal = read_temporal(data_dir, start_date, end_date)

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

    tmp = temporal.collect(streaming=True)
    logging.debug(f"Pre join = {tmp}")
    del tmp
    gc.collect()
    logging.debug("Joining tables")

    df: pl.DataFrame = (
        temporal.join(oil, on="date", how="left")
        .join(store_info, on="store_nbr", how="left")
        .join(items, on="item_nbr", how="left")
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
    logging.debug(f"Post join {df}")
    df.write_parquet(f"{data_dir}/df.parquet")
    return df


def read_temporal(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
) -> pl.LazyFrame:
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
        .pipe(downcast_dataframe)
        .sort("date", "traj_id")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
        .upsample("date", every="1d", by="traj_id")
        .fill_null(strategy="forward")
        .with_columns(pl.col("unit_sales").log())
        .rename({"unit_sales": "log_sales"})
    )

    return downcast_dataframe(temporal, streaming=True)


def downcast_dataframe(df: pl.DataFrame | pl.LazyFrame, streaming: bool = False) -> pl.LazyFrame:
    columns = df.columns

    df = df.with_columns(
        [pl.col(i).cast(_COLUMN_TO_DTYPE[i]) for i in columns if i in _COLUMN_TO_DTYPE]
    )

    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=streaming)

    df = df.shrink_to_fit(in_place=True).rechunk()
    return df.lazy()


# ------------- preprocessing ---------------


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    # In contrast to electricity for favorita we don't group before training StandardScaler
    target_scaler = StandardScaler()
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())
    # label_encoders["traj_id"].fit(df["traj_id"].to_numpy())
    target_scaler.fit(df[_TARGETS].to_numpy().reshape(-1, 1))
    for i in _REAL_INPUTS:
        real_scalers[i].fit(df[i].to_numpy().reshape(-1, 1))

    for i in tqdm(_CATEGORICAL_INPUTS, desc="Fitting label encoders"):
        label_encoders[i].fit(df[i].to_numpy())

    return {
        "real": dict(**real_scalers),
        "target": target_scaler,
        "categorical": dict(**label_encoders),
    }


def apply_preprocessor(
    df: pl.DataFrame,
    preprocessor: PreprocessorDict,
) -> pl.DataFrame:
    lf = df.lazy()

    log_sales = preprocessor["target"].transform(df[_TARGETS].to_numpy().reshape(-1, 1))
    lf = lf.drop("log_sales").with_columns(log_sales=pl.lit(log_sales))

    for i in tqdm(_REAL_INPUTS):
        x = df[i].to_numpy().reshape(-1, 1)
        x = preprocessor["real"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i))

    for i in tqdm(_CATEGORICAL_INPUTS):
        x = df[i].to_numpy()
        x = preprocessor["categorical"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i))

    df = lf.collect().shrink_to_fit(in_place=True).rechunk()
    return df


def convert_dataframe_to_tf_dataset(
    df: pl.DataFrame,
    preprocessor: PreprocessorDict,
    total_time_steps: int,
) -> tf.data.Dataset:
    df = apply_preprocessor(df, preprocessor)
    time_series = time_series_dataset_from_dataframe(
        df,
        inputs=_REAL_INPUTS + _CATEGORICAL_INPUTS,
        targets=_TARGETS,
        total_time_steps=total_time_steps,
        id_column="traj_id",
    )
    return time_series
