from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Tuple, TypedDict, Callable

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from tqdm.auto import tqdm
from jaxtyping import Float, Array

from temporal_fusion_transformer.src.experiments.base import (
    DataPreprocessorBase,
    MultiHorizonTimeSeriesDataset,
    TrainerBase,
)
from temporal_fusion_transformer.src.experiments.config import get_config
from temporal_fusion_transformer.src.experiments.util import (
    deserialize_preprocessor,
    time_series_dataset_from_dataframe,
    persist_dataset,
)

if TYPE_CHECKING:
    import polars as pl
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    from temporal_fusion_transformer.src.config_dict import ConfigDict
    from temporal_fusion_transformer.src.training.metrics import MetricContainer
    from temporal_fusion_transformer.src.training.training import DeviceTypeT
    from temporal_fusion_transformer.src.training.training_lib import (
        TrainStateContainer,
    )


try:
    import polars as pl
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ModuleNotFoundError as ex:
    logging.warning(ex)


# ------ We export class based API for convenience ------------


class Electricity(MultiHorizonTimeSeriesDataset):
    def __init__(
        self,
        validation_boundary: datetime = datetime(2014, 8, 8),
        test_boundary: int = datetime(2014, 9, 1),
        split_overlap_days: int = 7,
        cutoff_days: Tuple[datetime, datetime] = (datetime(2014, 1, 1), datetime(2014, 9, 8)),
    ):
        """
        References
        ----------

        - https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

        Parameters
        ----------
        validation_boundary:
            Timestamp, after which data is considered validation split.
        test_boundary:
            Timestamp, after which data is considered test split.
        split_overlap_days:
            Number of days, which are overlapped between splits.
        cutoff_days:
            Tuple of start and end dates, before/after which data is not included.
        """
        total_time_steps = get_config("electricity").total_time_steps
        self.validation_boundary = validation_boundary
        self.test_boundary = test_boundary
        self.cutoff_days = cutoff_days
        self.total_time_steps = total_time_steps
        self.split_overlap_days = split_overlap_days

    @property
    def trainer(self) -> Trainer:
        return Trainer()

    def make_dataset(
        self, data_dir: str, persist: bool = False, save_dir: str | None = None
    ) -> None | Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessor]:
        if save_dir is None:
            save_dir = data_dir
        training_ds, validation_ds, test_df, preprocessor = make_dataset(
            data_dir,
            validation_boundary=self.validation_boundary,
            test_boundary=self.test_boundary,
            total_time_steps=self.total_time_steps,
            split_overlap_days=self.split_overlap_days,
            cutoff_days=self.cutoff_days,
        )
        if persist:
            persist_dataset(training_ds, validation_ds, test_df, preprocessor.preprocessor, save_dir)
        else:
            return training_ds, validation_ds, test_df, preprocessor


class DataPreprocessor(DataPreprocessorBase):
    def __init__(self, preprocessor: PreprocessorDict, total_time_steps: int | None = None):
        if total_time_steps is None:
            total_time_steps = get_config("electricity").total_time_steps
        self.preprocessor = preprocessor
        self.total_time_steps = total_time_steps

    @staticmethod
    def load(file_name: str) -> DataPreprocessorBase:
        preprocessor = deserialize_preprocessor(file_name)
        return DataPreprocessor(preprocessor)

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        return apply_preprocessor(df, self.preprocessor)

    def convert_dataframe_to_tf_dataset(self, df: pl.DataFrame) -> tf.data.Dataset:
        return time_series_dataset_from_dataframe(
            df,
            inputs=_INPUTS,
            targets=_TARGETS,
            id_column=_ID_COLUMN,
            preprocess_fn=self.apply,
            total_time_steps=self.total_time_steps,
        )

    def inverse_transform(self, x_batch: Float[Array, "batch n"], y_batch: Float["batch 1"]) -> pl.DataFrame:
        # see config.py for indexes
        # 1 -> month
        # 2 -> day
        # 3 -> hour
        # 4 ->  day of week
        # 5 -> id
        encoded_ids = x_batch[..., 5]
        ids = self.preprocessor["categorical"]["id"].inverse_transform(encoded_ids)
        day = self.preprocessor["categorical"]["day"].inverse_transform(x_batch[..., 2])
        hour = self.preprocessor["categorical"]["hour"].inverse_transform(x_batch[..., 3])
        day_of_week = self.preprocessor["categorical"]["day_of_week"].inverse_transform(x_batch[..., 4])

        year_encoded = x_batch[..., 0]

        df_list = []

        for i, id_i in enumerate(ids):
            idx = np.argwhere(encoded_ids == id_i)
            day_i = day[idx]
            hour_i = day_i

            df = pl.DataFrame().with_cols(
                id=ids[i],
            )


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
        from temporal_fusion_transformer.src.training.training import train
        from temporal_fusion_transformer.src.training.training_lib import load_dataset

        data_config = get_config("electricity")

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
            save_path=save_path,
        )

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
        device_type: DeviceTypeT = "gpu",
        prefetch_buffer_size: int = 0,
    ) -> Tuple[Tuple[MetricContainer, MetricContainer], TrainStateContainer]:
        from temporal_fusion_transformer.src.training.training import train_distributed
        from temporal_fusion_transformer.src.training.training_lib import load_dataset

        data_config = get_config("electricity")
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
            tensorboard_logdir="tensorboard/electricity/",
            verbose=verbose,
        )


# -------------- actual implementations -----------------

_ID_COLUMN = "id"
_REAL_INPUTS = ["year"]
_CATEGORICAL_INPUTS = ["month", "day", "hour", "day_of_week"]
_INPUTS = _REAL_INPUTS + _CATEGORICAL_INPUTS + [_ID_COLUMN]
_TARGETS = ["power_usage"]


if TYPE_CHECKING:

    class CategoricalPreprocessorDict(TypedDict):
        day: LabelEncoder
        day_of_week: LabelEncoder
        hour: LabelEncoder
        id: LabelEncoder
        month: LabelEncoder

    class PreprocessorDict(TypedDict):
        """
        real and target have keys [MT_001 ... MT_370]

        """

        real: Mapping[str, StandardScaler]
        target: Mapping[str, StandardScaler]
        categorical: CategoricalPreprocessorDict


def make_dataset(
    data_dir: str,
    validation_boundary: datetime = datetime(2014, 8, 8),
    test_boundary: datetime = datetime(2014, 9, 1),
    split_overlap_days: int = 7,
    cutoff_days: Tuple[datetime, datetime] = (datetime(2014, 1, 1), datetime(2014, 9, 8)),
    total_time_steps: int = 192,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, DataPreprocessor]:
    convert_to_parquet(data_dir)
    df = read_parquet(data_dir, cutoff_days=cutoff_days)
    logging.info(f"{df.columns = }")
    preprocessor = train_preprocessor(df)
    training_df, validation_df, test_df = split_data(df, validation_boundary, test_boundary, split_overlap_days)

    make_dataset_fn: Callable[[pl.DataFrame], tf.data.Dataset] = partial(
        time_series_dataset_from_dataframe,
        inputs=_INPUTS,
        targets=_TARGETS,
        total_time_steps=total_time_steps,
        id_column=_ID_COLUMN,
        preprocess_fn=partial(apply_preprocessor, preprocessor=preprocessor),
    )
    training_time_series: tf.data.Dataset = make_dataset_fn(training_df)
    validation_time_series: tf.data.Dataset = make_dataset_fn(validation_df)
    return training_time_series, validation_time_series, test_df, DataPreprocessor(preprocessor)


def convert_to_parquet(data_dir: str):
    import polars as pl

    # FIXME: save in tmp dir

    if Path(f"{data_dir}/LD2011_2014.parquet").exists():
        logging.info(f"Found {data_dir}/LD2011_2014.parquet locally, will skip download.")
        return
    Path(data_dir).mkdir(exist_ok=True)
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


def read_parquet(data_dir: str, cutoff_days: Tuple[datetime, datetime]) -> pl.DataFrame:
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
                    pl.col("timestamp").dt.year().alias("year").cast(pl.UInt16),
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
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
        test_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
    )


# ------------------------- preprocessing and co ----------------


def train_preprocessor(df: pl.DataFrame) -> PreprocessorDict:
    target_scalers = defaultdict(lambda: StandardScaler())
    real_scalers = defaultdict(lambda: StandardScaler())
    label_encoders = defaultdict(lambda: LabelEncoder())

    label_encoders["id"].fit(df["id"].to_numpy())

    for i, sub_df in tqdm(df.groupby("id"), desc="Training scalers", total=370):
        target_scalers[i].fit(df[_TARGETS].to_numpy(order="c"))
        real_scalers[i].fit(df[_REAL_INPUTS].to_numpy(order="c"))

    for i in tqdm(_CATEGORICAL_INPUTS, desc="Fitting label encoders"):
        label_encoders[i].fit(df[i].to_numpy())

    return {"real": dict(**real_scalers), "target": dict(**target_scalers), "categorical": dict(**label_encoders)}


def apply_preprocessor(
    df: pl.DataFrame,
    preprocessor: PreprocessorDict,
) -> pl.DataFrame:
    # We don't declare them as constants, to avoid depending on external state.
    lf_list = []

    for i, sub_df in tqdm(df.groupby("id"), total=370, desc="Applying scalers..."):
        sub_df: pl.DataFrame
        sub_lf: pl.LazyFrame = sub_df.lazy()

        x_real = df[_REAL_INPUTS].to_numpy(order="c")
        x_target = df[_TARGETS].to_numpy(order="c")

        x_real = preprocessor["real"][i].transform(x_real)
        x_target = preprocessor["target"][i].transform(x_target)

        sub_lf = sub_lf.with_columns(
            [pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_real, _REAL_INPUTS)]
        ).with_columns(pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_target, _TARGETS))
        lf_list.append(sub_lf)

    df: pl.DataFrame = pl.concat(lf_list).collect()

    for i in tqdm(_CATEGORICAL_INPUTS, desc="Applying label encoders..."):
        x = df[i].to_numpy()
        x = preprocessor["categorical"][i].transform(x)
        df = df.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Int8))

    ids = preprocessor["categorical"]["id"].transform(df["id"].to_numpy())
    df = df.drop("id").with_columns(id=pl.lit(ids).cast(pl.UInt16)).shrink_to_fit(in_place=True).rechunk()
    df = df.shrink_to_fit(in_place=True).rechunk()
    return df


# -------------------- inference ---------------


def restore_timestamps(arr) -> List[List[datetime]]:
    # Indexes are taken from `config.py`
    year = arr[..., 0]
    month = arr[..., 1]
    day = arr[..., 2]
    hour = arr[..., 3]

    batch_size = arr.shape[0]

    timestamps_batch = []

    for i in range(batch_size):
        timestamps = [
            datetime(year=int(y), month=int(m), day=int(d), hour=int(h))
            for y, m, d, h in zip(year[i], month[i], day[i], hour[i])
        ]
        timestamps_batch.append(timestamps)

    return timestamps_batch
