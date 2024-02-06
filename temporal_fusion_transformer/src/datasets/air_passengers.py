from __future__ import annotations

import logging

import polars as pl

from temporal_fusion_transformer.src.datasets.base import (
    PreprocessorBase,
)

log = logging.getLogger(__name__)


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
