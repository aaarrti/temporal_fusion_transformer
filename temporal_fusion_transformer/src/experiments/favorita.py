from __future__ import annotations

import gc
import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from importlib import util
from typing import TYPE_CHECKING, Callable, List, Literal, Mapping, Tuple, TypedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from tqdm.auto import tqdm

from temporal_fusion_transformer.src.experiments.base import (
    DataPreprocessorBase,
    MultiHorizonTimeSeriesDataset,
    TrainerBase,
)
from temporal_fusion_transformer.src.experiments.configs import (
    fixed_parameters,
    hyperparameters,
)
from temporal_fusion_transformer.src.experiments.util import (
    deserialize_preprocessor,
    persist_dataset,
    time_series_dataset_from_dataframe,
)

if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf

    from temporal_fusion_transformer.src.config_dict import ConfigDict, ModelConfig
    from temporal_fusion_transformer.src.lib_types import (
        DeviceTypeT,
        HooksT,
        PredictFn,
        TrainingResult,
    )

try:
    import polars as pl
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ModuleNotFoundError as ex:
    logging.warning(ex)


class Favorita(MultiHorizonTimeSeriesDataset):
    trainer = property(lambda self: Trainer())

    def __init__(
        self,
        start_date: datetime | None = datetime(2016, 1, 1),
        end_date: datetime | None = datetime(2016, 6, 1),
        validation_boundary: datetime = datetime(2016, 4, 1),
    ):
        config = fixed_parameters.get_config("favorita")
        self.start_date = start_date
        self.end_date = end_date
        self.validation_boundary = validation_boundary
        self.total_time_steps = config.total_time_steps
        self.num_encoder_steps = config.num_encoder_steps

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
            cache_dir=save_dir,
        )
        if save_dir is not None:
            persist_dataset(training_ds, validation_ds, test_df, preprocessor.preprocessor, save_dir)
        else:
            return training_ds, validation_ds, test_df, preprocessor

    def convert_to_parquet(self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True):
        convert_to_parquet(download_dir, output_dir, delete_processed=delete_processed)

    def reload_preprocessor(self, filename: str) -> DataPreprocessor:
        return DataPreprocessor.load(filename)

    def reload_model(
        self, filename: str, config: ModelConfig, jit_module: bool, return_attention: bool = True
    ) -> PredictFn:
        from temporal_fusion_transformer.src.inference.util import reload_model

        if config is None:
            config = hyperparameters.get_config("favorita")

        return reload_model(
            filename,
            model_config=config,
            data_config=fixed_parameters.get_config("favorita"),
            jit_module=jit_module,
            return_attention=return_attention,
        )


class DataPreprocessor(DataPreprocessorBase):
    __slots__ = ["preprocessor", "total_time_steps"]

    def __init__(self, preprocessor: PreprocessorDict):
        self.preprocessor = preprocessor
        self.total_time_steps = fixed_parameters.get_config("favorita").total_time_steps

    @staticmethod
    def load(file_name: str) -> DataPreprocessorBase:
        preprocessor = deserialize_preprocessor(file_name)
        return DataPreprocessor(preprocessor)

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return apply_preprocessor(df, self.preprocessor)

    def convert_dataframe_to_tf_dataset(self, df: pl.DataFrame) -> tf.data.Dataset:
        return convert_dataframe_to_tf_dataset(df, self.preprocessor, self.total_time_steps)

    def inverse_transform(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        pass

    def restore_timestamps(self, df: pl.DataFrame) -> List[datetime]:
        pass


class Trainer(TrainerBase):
    def run(
        self,
        *,
        data_dir: str,
        batch_size: int,
        config: ConfigDict | Literal["auto"] = "auto",
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        verbose: bool = True,
        hooks: HooksT = "auto",
    ) -> TrainingResult:
        from temporal_fusion_transformer.src.training.training import train
        from temporal_fusion_transformer.src.training.training_lib import load_dataset

        if config == "auto":
            config = hyperparameters.get_config("favorita")

        data_config = fixed_parameters.get_config("favorita")

        data = load_dataset(
            data_dir,
            batch_size,
            prng_seed=config.prng_seed,
            dtype=jnp.float16 if mixed_precision else jnp.float32,
            shuffle_buffer_size=config.shuffle_buffer_size,
            num_encoder_steps=data_config.num_encoder_steps,
        )

        return train(
            data=data,
            config=config,
            data_config=data_config,
            mixed_precision=mixed_precision,
            epochs=epochs,
            verbose=verbose,
            hooks=hooks,
        )

    def run_distributed(
        self,
        *,
        data_dir: str,
        batch_size: int,
        device_type: DeviceTypeT,
        config: ConfigDict | Literal["auto"] = "auto",
        epochs: int = 1,
        mixed_precision: bool = False,
        jit_module: bool = False,
        verbose: bool = True,
        prefetch_buffer_size: int = 0,
        hooks: HooksT = "auto",
    ) -> TrainingResult:
        from temporal_fusion_transformer.src.training.training import train_distributed
        from temporal_fusion_transformer.src.training.training_lib import load_dataset

        if config == "auto":
            config = hyperparameters.get_config("favorita")

        data_config = fixed_parameters.get_config("favorita")
        num_devices = jax.device_count()

        data = load_dataset(
            data_dir,
            batch_size * num_devices,
            prng_seed=config.prng_seed,
            dtype=jnp.float16 if mixed_precision else jnp.float32,
            shuffle_buffer_size=config.shuffle_buffer_size,
            num_encoder_steps=data_config.num_encoder_steps,
        )

        return train_distributed(
            data=data,
            config=config,
            data_config=data_config,
            mixed_precision=mixed_precision,
            epochs=epochs,
            device_type=device_type,
            prefetch_buffer_size=prefetch_buffer_size,
            verbose=verbose,
            hooks=hooks,
        )


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
    cache_dir: str | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessor]:
    df = read_parquet(data_dir, start_date, end_date, cache_dir=cache_dir)
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
    return training_ds, validation_ds, test_df, DataPreprocessor(preprocessor)


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

    training_df: pl.DataFrame = lf.filter(pl.col("date").lt(validation_boundary).over("traj_id")).collect()
    validation_df = df.filter(pl.col("date").ge(validation_boundary).over("traj_id")).filter(
        pl.col("date").lt(test_boundary).over("traj_id")
    )
    test_df = df.filter(pl.col("date").ge(test_boundary).over("traj_id"))

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
    cache_dir: str | None = None,
) -> pl.DataFrame:
    if cache_dir is None:
        cache_dir = data_dir

    if Path(f"{cache_dir}/joined_df.parquet").exists():
        logging.info("Found joined_df.parquet, will re-use it.")
        return pl.read_parquet(f"{cache_dir}/joined_df.parquet")

    temporal = read_temporal(data_dir, start_date, end_date)
    tmp = temporal.collect(streaming=True)

    store_info = pl.scan_parquet(f"{data_dir}/stores.parquet").pipe(downcast_dataframe)
    items = pl.scan_parquet(f"{data_dir}/items.parquet").pipe(downcast_dataframe)
    transactions = pl.scan_parquet(f"{data_dir}/transactions.parquet").pipe(downcast_dataframe)
    oil = pl.scan_parquet(f"{data_dir}/oil.parquet").rename({"dcoilwtico": "oil_price"}).pipe(downcast_dataframe)
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
        .with_columns(pl.col("oil_price").fill_null(strategy="forward").over("traj_id"))
        .join(store_info, on="store_nbr")
        .join(items, on="item_nbr")
        .join(transactions, on=["store_nbr", "date"])
        .with_columns(pl.col("transactions").fill_null(strategy="forward").over("traj_id"))
        .join(national_holidays, on="date", how="left")
        .join(regional_holidays, on=["date", "state"], how="left")
        .join(local_holidays, on=["date", "city"], how="left")
        .with_columns(
            [
                pl.col("national_hol").fill_null(""),
                pl.col("regional_hol").fill_null(""),
                pl.col("local_hol").fill_null(""),
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
        .with_columns([pl.format("{}_{}", "store_nbr", "item_nbr").alias("traj_id")])
        .pipe(remove_returns_data)
        .pipe(downcast_dataframe)
        .sort("date", "traj_id")
        .collect(streaming=True)
        .shrink_to_fit(in_place=True)
        .rechunk()
        .upsample("date", every="1d", by="traj_id")
        .lazy()
        .with_columns(
            [pl.col(i).fill_null(strategy="forward").over("traj_id") for i in ["store_nbr", "item_nbr", "onpromotion"]]
        )
        .with_columns(pl.col("open").fill_null(0))
        .with_columns(pl.col("unit_sales").log())
        .rename({"unit_sales": "log_sales"})
        .with_columns(pl.col("log_sales").fill_null(strategy="forward").over("traj_id"))
    )

    return downcast_dataframe(temporal, streaming=True)


def downcast_dataframe(df: pl.DataFrame | pl.LazyFrame, streaming: bool = False) -> pl.LazyFrame:
    columns = df.columns

    df = df.with_columns([pl.col(i).cast(_COLUMN_TO_DTYPE[i]) for i in columns if i in _COLUMN_TO_DTYPE])

    if isinstance(df, pl.LazyFrame):
        df = df.collect(streaming=streaming)

    df = df.shrink_to_fit(in_place=True).rechunk()
    return df.lazy()


# ------------- preprocessing ---------------


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    # In contrast to favorita for favorita we don't group before training StandardScaler
    target_scaler = StandardScaler()
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())
    # label_encoders["traj_id"].fit(df["traj_id"].to_numpy())
    target_scaler.fit(df[_TARGETS].to_numpy(order="c"))
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
    lf = lf.drop("log_sales").with_columns(log_sales=pl.lit(log_sales.reshape(-1)).cast(pl.Float32))

    for i in tqdm(_REAL_INPUTS):
        x = df[i].to_numpy().reshape(-1, 1)
        x = preprocessor["real"][i].transform(x)
        lf = lf.drop(i).with_columns(pl.lit(x.reshape(-1)).alias(i))

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
