from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler

from temporal_fusion_transformer.src.datasets.preprocessor import PreprocessorBase

_CATEGORICAL_INPUTS = ["month", "day", "hour", "day_of_week"]

log = logging.getLogger(__name__)


class Preprocessor(PreprocessorBase):
    
    def __init__(self, state: dict[str, dict[str, ...]] | None = None):
        if state is None:
            state = {
                "year": defaultdict(StandardScaler),
                "target": defaultdict(StandardScaler),
                "categorical": defaultdict(LabelEncoder)
            }
        super().__init__(state)
        
    @property
    def year(self) -> dict[str, StandardScaler]:
        """
        
        Returns
        -------
        
        year: dict[id, StandardScaler]

        """
        return self.state['year']
    
    @property
    def target(self) -> dict[str, StandardScaler]:
        """
        
        Returns
        -------
        
        target: dict[id, StandardScaler]

        """
        return self.state['target']
    
    @property
    def categorical(self) -> dict[str, LabelEncoder]:
        """
        
        Returns
        -------
        
        categorical: dict[feature, LabelEncoder]

        """
        return self.state['categorical']
    
    def group_mapper(self, group_df: pl.DataFrame) -> pl.DataFrame:
        group_id = group_df["id"][0]
        
        new_df = group_df.with_columns(
            pl.col('year').map_batches(
                _make_real_mapper(self.year[group_id])
            ),
            pl.col('y').map_batches(
                _make_real_mapper(self.target[group_id])
            )
        )
        
        return new_df
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.group_by("id")
            .map_groups(self.group_mapper)
            .with_columns(
                [
                    pl.col(i).map_batches(_make_categorical_mapper(self.categorical[i]))
                    for i in ('id', "month", "day", "hour", "day_of_week")
                ]
            )
            .shrink_to_fit(in_place=True)
            .rechunk()
            .select(
                'id',
                "ts",
                "year",
                "month",
                "day",
                "day_of_week",
                "hour",
                "y",
            )
        )
        return df
    
    def fit(self, df: pl.DataFrame):
        for i, sub_df in df.group_by(["id"]):
            i = i[0]
            self.target[i].fit(df.select('y').to_numpy(order="c"))
            self.year[i].fit(df.select('year').to_numpy(order="c"))
        
        for i in ('id', "month", "day", "hour", "day_of_week"):
            self.categorical[i].fit(df[i].to_numpy())
        
        self.state = {
            "target": dict(**self.target),
            "year": dict(**self.year),
            "categorical": dict(**self.categorical)
        }
    
    @staticmethod
    def to_array(dataframe: pl.DataFrame) -> np.ndarray:
        return (
            dataframe.select(
                'id',
                "year",
                "month",
                "day",
                "day_of_week",
                "hour",
                "y",
            ).to_numpy(order='c')
        )
        
    

def _make_categorical_mapper(encoder: LabelEncoder) -> Callable[[pl.Series], pl.Series]:
    def map_fn(s: pl.Series):
        return pl.Series(encoder.transform(s.to_numpy()))
    
    return map_fn


def _make_real_mapper(scaler: StandardScaler) -> Callable[[pl.Series], pl.Series]:
    def map_fn(s: pl.Series):
        arr = np.asarray(s).reshape((-1, 1))
        arr = scaler.transform(arr).reshape(-1)
        return pl.Series(arr)
    
    return map_fn
