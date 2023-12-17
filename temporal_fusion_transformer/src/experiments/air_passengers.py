import logging
from typing import TYPE_CHECKING, NoReturn

import keras_core as keras
import polars as pl
import tensorflow as tf
from keras_core import callbacks
from keras_core.layers import IntegerLookup, Normalization
from keras_core.mixed_precision import global_policy
from ml_collections import config_flags

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.experiments.base import (
    Experiment,
    MultiHorizonTimeSeriesDataset,
    Preprocessor,
)
from temporal_fusion_transformer.src.experiments.utils import (
    persist_dataset,
    time_series_dataset_from_dataframe,
)

log = logging.getLogger(__name__)
_CONFIG = config_flags.DEFINE_config_file(
    "air_passengers", default=__file__.replace("air_passengers.py", "config.py:air_passengers")
)


class AirPassengers(Experiment):
    @property
    def dataset_cls(self) -> type[MultiHorizonTimeSeriesDataset]:
        return AirPassengersDataset

    @property
    def preprocessor_cls(self) -> type[Preprocessor]:
        return AirPassengerPreprocessor

    @property
    def config(self) -> Config:
        return _CONFIG.value

    def train_model(
        self,
        data_dir: str = "data/air_passengers",
        batch_size: int = 16,
        epochs: int = 100,
        save_filename: str | None = None,
        **kwargs,
    ) -> keras.Model | None:
        from temporal_fusion_transformer.src import training

        config = self.config

        total_time_steps = config.data.total_time_steps

        dataset = training.load_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            encoder_steps=config.data.encoder_steps,
            dtype=global_policy().compute_dtype,
            element_spec=(
                tf.TensorSpec(shape=(total_time_steps, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(total_time_steps, 1), dtype=tf.float32),
            ),
            compression=None,
        )
        return training.train_model(
            dataset=dataset,
            epochs=epochs,
            save_filename=save_filename,
            config=config,
            training_callbacks=[
                callbacks.TerminateOnNaN(),
            ],
            **kwargs,
        )


# ----------------------------------------------------------------------------------------


class AirPassengersDataset(MultiHorizonTimeSeriesDataset):
    def __init__(self, validation_boundary: int = 1957, test_boundary: int = 1959):
        """
        Dataset contains data for years 1949 - 1960
        """
        self.validation_boundary = validation_boundary
        self.test_boundary = test_boundary
        self.config = _CONFIG.value
        self.preprocessor = AirPassengerPreprocessor()

    def convert_to_parquet(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError

    def make_dataset(
        self, data_dir: str, save_dir: str | None = None
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, Preprocessor] | None:
        df = (
            pl.read_csv(f"{data_dir}/AirPassengers.csv")
            # We need to append fake id to use as static embeddings
            .with_columns(id=0)
            .with_columns(pl.col("Month").str.to_date("%Y-%m"))
            .with_columns(pl.col("Month").dt.month_start())
            .sort("Month")
            .upsample("Month", every="1mo")
            .with_columns(
                pl.col("Month").dt.month().alias("month"), pl.col("Month").dt.year().alias("year")
            )
            .drop("Month")
            .rename({"#Passengers": "passengers"})
        )

        self.preprocessor.adapt(df)

        train_df = df.filter(pl.col("year").le(self.validation_boundary))
        validation_df = df.filter(pl.col("year").ge(self.validation_boundary)).filter(
            pl.col("year").le(self.test_boundary)
        )
        test_df = df.filter(pl.col("year").ge(self.test_boundary))

        log.info(
            f"Using {len(train_df)} samples for training, {len(validation_df)} samples for validation and {len(test_df)} for test"
        )

        train_ds = self.make_time_series(train_df)
        validation_ds = self.make_time_series(validation_df)

        if save_dir is not None:
            return persist_dataset(
                train_ds,
                validation_ds,
                test_df,
                self.preprocessor,
                save_dir,
                test_split_save_format="csv",
                compression=None,
            )
        else:
            return train_ds, validation_ds, test_df, self.preprocessor

    def make_time_series(self, ts: pl.DataFrame) -> tf.data.Dataset:
        return time_series_dataset_from_dataframe(
            ts,
            inputs=["id", "month", "year"],
            targets=["passengers"],
            id_column="id",
            total_time_steps=self.config.data.total_time_steps,
            preprocessor=self.preprocessor,
        )


# ----------------------------------------------------------------------------------------


class AirPassengerPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__(
            {
                "real": {
                    "year": Normalization(),
                },
                "target": {
                    "passengers": Normalization(),
                },
                "categorical": {"month": IntegerLookup()},
            }
        )

    def adapt(self, df: pl.DataFrame):
        self.state["real"]["year"].adapt(
            tf.data.Dataset.from_tensor_slices(df.select("year").to_numpy(order="c"))
        )
        self.state["target"]["passengers"].adapt(
            tf.data.Dataset.from_tensor_slices(df.select("passengers").to_numpy(order="c"))
        )
        self.state["categorical"]["month"].adapt(
            tf.data.Dataset.from_tensor_slices(df.select("passengers").to_numpy(order="c"))
        )
        self.built = True

    def call(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        year = self.state["real"]["year"](df.select("year").to_numpy(order="c")).numpy().reshape(-1)
        month = (
            self.state["categorical"]["month"](df.select("month").to_numpy(order="c"))
            .numpy()
            .reshape(-1)
        )
        passengers = (
            self.state["target"]["passengers"](df.select("passengers").to_numpy(order="c"))
            .numpy()
            .reshape(-1)
        )

        return pl.DataFrame(
            {"year": year, "month": month, "passengers": passengers, "id": df["id"]}
        )
