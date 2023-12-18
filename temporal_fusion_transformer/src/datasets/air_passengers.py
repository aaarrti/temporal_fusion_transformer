from __future__ import annotations

import dataclasses
import logging
import pickle
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.datasets.base import (
    MultiHorizonTimeSeriesDataset,
    PreprocessorBase,
    SplitSpec,
)
from temporal_fusion_transformer.src.datasets.utils import (
    dataframe_from_time_series_dataset,
    time_series_dataset_from_dataframe,
    time_series_to_array,
)

log = logging.getLogger(__name__)


class AirPassengersDataset(MultiHorizonTimeSeriesDataset):
    def __init__(self, config: Config):
        """
        Dataset contains data for years 1949 - 1960
        """
        self.config = config
        self.preprocessor = AirPassengerPreprocessor()

    @staticmethod
    def read_df(data_dir: str) -> pl.DataFrame:
        return (
            pl.read_csv(f"{data_dir}/AirPassengers.csv")
            # We need to append fake id to use as static embeddings
            .with_columns(id=0)
            .with_columns(pl.col("Month").str.to_date("%Y-%m"))
            .with_columns(pl.col("Month").dt.month_start())
            .sort("Month")
            .upsample("Month", every="1mo")
            .rename({"#Passengers": "passengers"})
        )

    def make_dataset(self, data_dir: str):
        df = self.read_df(data_dir)
        self.preprocessor.fit(self.split_time_column(df))

        split_spec = compute_split_spec(self.config, df)

        train_df = df.filter(pl.col("Month").le(split_spec.train_end))
        validation_df = df.filter(pl.col("Month").ge(split_spec.val_start)).filter(
            pl.col("Month").le(split_spec.val_end)
        )
        test_df = df.filter(pl.col("Month").ge(split_spec.test_start))

        train_df = self.split_time_column(train_df)
        validation_df = self.split_time_column(validation_df)
        test_df = self.split_time_column(test_df)

        log.info(
            f"Using {len(train_df)} samples for training, {len(validation_df)} samples for validation and {len(test_df)} for test"
        )

        train_ds = self.make_time_series(train_df)
        validation_ds = self.make_time_series(validation_df)
        return train_ds, validation_ds, test_df, self.preprocessor

    def make_time_series(self, ts: pl.DataFrame) -> tf.data.Dataset:
        return time_series_dataset_from_dataframe(
            ts,
            inputs=["id", "month", "year"],
            targets=["passengers"],
            id_column="id",
            total_time_steps=self.config.total_time_steps,
            preprocessor=self.preprocessor.transform,
        )

    @staticmethod
    def split_time_column(df: pl.DataFrame):
        return df.with_columns(
            pl.col("Month").dt.month().alias("month"), pl.col("Month").dt.year().alias("year")
        ).drop("Month")

    def plot_dataset_splits(self, data_dir: str, *args):
        df = self.read_df(data_dir)
        split_spec = compute_split_spec(self.config, df)

        df = self.split_time_column(df)

        ts, y = extract_time_steps_and_targets(df)

        plt.plot(ts, y, marker="o", markersize=4, color="gray")

        for (k, v), c in zip(
            dataclasses.asdict(split_spec).items(), ("blue", "green", "red", "magenta")
        ):
            plt.axvline(x=v, color=c, linestyle="dashed", label=k)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def convert_to_parquet(
        self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True
    ):
        log.debug("not implemented (also not needed).")


class AirPassengerPreprocessor(PreprocessorBase):
    def __init__(
        self, year: StandardScaler | None = None, passengers: StandardScaler | None = None
    ):
        if year is None:
            year = StandardScaler()

        if passengers is None:
            passengers = StandardScaler()

        self.year = year
        self.passengers = passengers

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        def map_passengers(ps: pl.Series):
            return pl.Series(self.passengers.transform(ps.to_numpy().reshape(-1, 1)).reshape(-1))

        def map_year(y: pl.Series):
            return pl.Series(self.year.transform(y.to_numpy().reshape(-1, 1)).reshape(-1))

        return df.with_columns(
            pl.col("passengers").map_batches(map_passengers, return_dtype=pl.Float64),
            pl.col("year").map_batches(map_year, return_dtype=pl.Float64),
        )

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        def map_passengers(ps: pl.Series):
            return pl.Series(
                self.passengers.inverse_transform(ps.to_numpy().reshape(-1, 1)).reshape(-1)
            )

        def map_year(y: pl.Series):
            return pl.Series(self.year.inverse_transform(y.to_numpy().reshape(-1, 1)).reshape(-1))

        return df.with_columns(
            pl.col("passengers").map_batches(map_passengers, return_dtype=pl.Float32),
            pl.col("year").map_batches(map_year, return_dtype=pl.Float32),
        )

    def fit(self, df: pl.DataFrame):
        self.year.fit(df.select("year").to_numpy(order="c"))
        self.passengers.fit(df.select("passengers").to_numpy(order="c"))

    def save(self, dirname: str):
        joblib.dump(
            self.year, f"{dirname}/preprocessor.year.joblib", protocol=pickle.HIGHEST_PROTOCOL
        )
        joblib.dump(
            self.passengers,
            f"{dirname}/preprocessor.passengers.joblib",
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def load(dirname: str) -> AirPassengerPreprocessor:
        year = joblib.load(f"{dirname}/preprocessor.year.joblib")
        passengers = joblib.load(f"{dirname}/preprocessor.passengers.joblib")
        return AirPassengerPreprocessor(year, passengers)

    def __str__(self) -> str:
        return f"AirPassengerPreprocessor(year={str(self.year)}, passengers={str(self.passengers)})"

    def __repr__(self) -> str:
        return (
            f"AirPassengerPreprocessor(year={repr(self.year)}, passengers={repr(self.passengers)})"
        )


class AirPassengersInference:
    def __init__(self, config: Config):
        self.config = config

    def plot_predictions(
        self,
        df: pl.DataFrame,
        y_pred: np.ndarray,
        target_scaler: StandardScaler,
        history_df: pl.DataFrame | None = None,
    ):
        if history_df is not None:
            history_ts, history_y = extract_time_steps_and_targets(history_df)
            plt.plot(history_ts, history_y, label="historical data", marker="o", markersize=4)

        ts, y = extract_time_steps_and_targets(df)
        plt.plot(ts, y, label="ground truth", marker="o", markersize=4)

        # q_predictions = []

        for i, q in enumerate(self.config.quantiles):
            label = f"q({int(q*100)}%) prediction"
            q_prediction = target_scaler.inverse_transform(
                time_series_to_array(y_pred[..., i])
            ).reshape(-1)

            # q_predictions.append(q_prediction)
            plt.plot(
                ts[self.config.encoder_steps :], q_prediction, label=label, marker="o", markersize=4
            )

        # median = q_predictions[1]
        # percentile_10 = q_predictions[0]
        # percentile_90 = q_predictions[2]
        # Estimate mean and standard deviation assuming normal distribution
        # z_score = 1.96  # Z-score for a 95% confidence level

        # mean_estimate = median + (percentile_90 - percentile_10) / (2 * z_score)

        # plt.plot(
        #    ts[self.config.encoder_steps:], mean_estimate, label="mean_estimate", marker="o", markersize=4
        # )

        plt.legend()
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def dataframe_from_time_series_dataset(ds: tf.data.Dataset) -> pl.DataFrame:
        return dataframe_from_time_series_dataset(
            ds, inputs_mappings={"id": 0, "month": 1, "year": 2}, targets_mappings={"passengers": 0}
        )

    def prepare_model_input(self, df: pl.DataFrame) -> np.ndarray:
        ds = time_series_dataset_from_dataframe(
            df,
            ["id", "month", "year"],
            ["passengers"],
            self.config.total_time_steps,
            "id",
        )
        x = np.asarray([i for i, _ in ds.as_numpy_iterator()])
        return x


def compute_split_spec(config: Config, df: pl.DataFrame) -> SplitSpec:
    split_overlap = timedelta(days=31 * config.split_overlap)

    train_df_end = datetime(year=config.validation_boundary, month=1, day=1)

    val_df_start = datetime(config.validation_boundary, month=1, day=1) - split_overlap

    val_df_end = datetime(config.test_boundary, month=1, day=1) - split_overlap

    test_df_start = (
        df["Month"].max() - timedelta(days=31 * config.total_time_steps) - (split_overlap * 2)
    )
    return SplitSpec(
        train_end=train_df_end, val_end=val_df_end, val_start=val_df_start, test_start=test_df_start
    )


def extract_time_steps_and_targets(df: pl.DataFrame) -> tuple[list[datetime], list[float]]:
    ts = [datetime(year=int(y), month=int(m), day=1) for m, y in zip(df["month"], df["year"])]
    y = df["passengers"].to_list()

    return ts, y
