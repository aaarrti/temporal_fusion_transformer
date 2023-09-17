from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from datetime import datetime, timedelta
from glob import glob
from typing import TYPE_CHECKING, Literal, Mapping, Tuple, TypedDict, overload, Callable

import numpy as np
from absl import logging
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.experiments.base import (
    DataPreprocessorBase,
    MultiHorizonTimeSeriesDataset,
    TrainerBase,
)
from temporal_fusion_transformer.src.experiments.config import get_config
from temporal_fusion_transformer.src.experiments.util import (
    serialize_preprocessor,
    time_series_dataset_from_dataframe,
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

    @overload
    def make_dataset(self, data_dir: str, mode: Literal["persist"]) -> None:
        ...

    @overload
    def make_dataset(
        self, data_dir: str, mode: Literal["return"]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessor]:
        ...

    def make_dataset(
        self, data_dir: str, mode: Literal["persist", "return"] = "return"
    ) -> None | Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessorBase]:
        return make_dataset(
            data_dir=data_dir,
            mode=mode,
            start_date=self.start_date,
            end_date=self.end_date,
            validation_boundary=self.validation_boundary,
            total_time_steps=self.total_time_steps,
            num_encoder_steps=self.num_encoder_steps,
        )


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
    "day of month",
    "day of week",
    "national holiday" "regional hol",
    "local holiday",
    "onpromotion",
    "open",
]


@overload
def make_dataset(
    data_dir: str,
    start_date: datetime | None,
    end_date: datetime | None,
    validation_boundary: datetime,
    total_time_steps: int,
    num_encoder_steps: int,
    mode: Literal["return"],
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], PreprocessorDict]:
    ...


@overload
def make_dataset(
    data_dir: str,
    start_date: datetime | None,
    end_date: datetime | None,
    validation_boundary: datetime,
    total_time_steps: int,
    num_encoder_steps: int,
    mode: Literal["persist"],
) -> None:
    ...


def make_dataset(
    data_dir: str,
    start_date: datetime | None = datetime(2016, 1, 1),
    end_date: datetime | None = datetime(2016, 6, 1),
    validation_boundary: datetime = datetime(2016, 4, 1),
    total_time_steps: int = 120,
    num_encoder_steps: int = 90,
    mode: Literal["return", "persist"] = "return",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, PreprocessorDict]:
    convert_to_parquet(data_dir)

    df = read_parquet(data_dir, start_date, end_date)
    logging.debug(f"{df.describe() = }")
    preprocessor = train_preprocessor(df)
    training_df, validation_df, test_df = split_data(
        df, validation_boundary, total_time_steps=total_time_steps, num_encoder_steps=num_encoder_steps
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

    if mode == "return":
        return training_ds, validation_ds, test_df

    training_ds.save(f"{data_dir}/training", compression="GZIP")
    validation_ds.save(f"{data_dir}/validation", compression="GZIP")
    test_df.write_parquet(f"{data_dir}/test.parquet")
    serialize_preprocessor(preprocessor, data_dir)


def convert_to_parquet(data_dir: str):
    files = glob(f"{data_dir}/*.parquet")

    filenames = {i.rpartition("/")[-1] for i in files}
    missing_files = list(set(_REQUIRED_FILES).difference(filenames))

    if len(missing_files) == 0:
        logging.info(f"Found {files} locally.")
        return

    files = glob(f"{data_dir}/*.csv")
    for f in tqdm(files, desc="Converting to parquet"):
        f: str
        pl.scan_csv(f, try_parse_dates=True).sink_parquet(f.replace("csv", "parquet"))
        os.remove(f)


def split_data(
    df: pl.DataFrame, validation_boundary=datetime(2016, 4, 1), total_time_steps: int = 120, num_encoder_steps: int = 90
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

    return training_df, validation_df, test_df


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


# ------------- preprocessing ---------------


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    # In contrast to electricity for favorita we don't group before training StandardScaler
    target_scaler = StandardScaler()
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())
    # label_encoders["traj_id"].fit(df["traj_id"].to_numpy())
    target_scaler.fit(df["log_sales"].to_numpy().reshape(1, -1))
    for i in _REAL_INPUTS:
        real_scalers[i].fit(df[i].to_numpy().reshape(1, -1))

    for i in tqdm(_CATEGORICAL_INPUTS, desc="Fitting label encoders"):
        label_encoders[i].fit(df[i].to_numpy())

    return {"real": dict(**real_scalers), "target": target_scaler, "categorical": dict(**label_encoders)}


def apply_preprocessor(
    df: pl.DataFrame,
    preprocessor: PreprocessorDict,
) -> pl.DataFrame:
    lf = df.lazy()

    log_sales = preprocessor["target"].transform(df["log_sales"].to_numpy())
    lf = lf.drop("log_sales").with_columns(log_sales=pl.lit(log_sales).cast(pl.Float32))

    for i in tqdm(_REAL_INPUTS):
        x = df[i].to_numpy()
        x = preprocessor["real"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Float32))

    for i in tqdm(_CATEGORICAL_INPUTS):
        x = df[i].to_numpy()
        x = preprocessor["categorical"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Int8))

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
        targets=["log_sales"],
        total_time_steps=total_time_steps,
        id_column="traj_id",
    )
    return time_series
