from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
        self.preprocessor = AirPassengerPreprocessor(
            {"year": StandardScaler()}, {"passengers": StandardScaler()}, {"month": LabelEncoder()}
        )

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
        validation_df = df.filter(pl.col("Month").ge(split_spec.validation_start)).filter(
            pl.col("Month").le(split_spec.validation_end)
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
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        def map_passengers(ps: pl.Series):
            return pl.Series(
                self.target["passengers"].transform(ps.to_numpy().reshape(-1, 1)).reshape(-1)
            )

        def map_year(y: pl.Series):
            return pl.Series(self.real["year"].transform(y.to_numpy().reshape(-1, 1)).reshape(-1))

        def map_month(y: pl.Series):
            return pl.Series(self.categorical["month"].transform(y.to_numpy()))

        return df.with_columns(
            pl.col("passengers").map_batches(map_passengers, return_dtype=pl.Float64),
            pl.col("year").map_batches(map_year, return_dtype=pl.Float64),
            pl.col("month").map_batches(map_month),
        )

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        def map_passengers(ps: pl.Series):
            return pl.Series(
                self.target["passengers"]
                .inverse_transform(ps.to_numpy().reshape(-1, 1))
                .reshape(-1)
            )

        def map_year(y: pl.Series):
            return pl.Series(
                self.real["year"].inverse_transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
            )

        def map_month(y: pl.Series):
            return pl.Series(
                self.categorical["month"].inverse_transform(y.to_numpy().astype("int")).tolist()
            )

        return df.with_columns(
            pl.col("passengers").map_batches(map_passengers, return_dtype=pl.Float32),
            pl.col("year").map_batches(map_year, return_dtype=pl.Float32),
            pl.col("month").map_batches(map_month, return_dtype=pl.Int64),
        )

    def fit(self, df: pl.DataFrame):
        self.real["year"].fit(df.select("year").to_numpy(order="c"))
        self.target["passengers"].fit(df.select("passengers").to_numpy(order="c"))
        self.categorical["month"].fit(df["month"].to_numpy())


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
            plt.plot(
                history_ts,
                history_y,
                label="historical data",
                marker="o",
                markersize=4,
                color="darkblue",
            )

        ts, y = extract_time_steps_and_targets(df)
        plt.plot(ts, y, label="ground truth", marker="o", markersize=4)

        for i, q in enumerate(self.config.quantiles):
            label = f"q({int(q*100)}%) prediction"
            q_prediction = target_scaler.inverse_transform(
                time_series_to_array(y_pred[..., i])
            ).reshape(-1)
            plt.plot(
                ts[self.config.encoder_steps :], q_prediction, label=label, marker="o", markersize=4
            )

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
        train_end=train_df_end,
        validation_end=val_df_end,
        validation_start=val_df_start,
        test_start=test_df_start,
    )


def extract_time_steps_and_targets(df: pl.DataFrame) -> tuple[list[datetime], list[float]]:
    ts = [datetime(year=int(y), month=int(m), day=1) for m, y in zip(df["month"], df["year"])]
    y = df["passengers"].to_list()

    return ts, y
