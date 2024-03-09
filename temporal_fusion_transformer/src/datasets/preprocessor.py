from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

import joblib
import polars as pl
from sklearn.preprocessing import FunctionTransformer


class PreprocessorBase(ABC):

    def __init__(self, state: dict[str, ...]):
        self.state = state

    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def save(self, dirname: str):
        joblib.dump(
            self.state,
            f"{dirname}/preprocessor.joblib",
            compress=3,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def load(cls, dirname: str) -> PreprocessorBase:
        state = joblib.load(f"{dirname}/preprocessor.joblib")
        return cls(state)

    def __repr__(self) -> str:
        return repr({k: repr(v) for k, v in self.state.items()})


def MonthNormalizer():
    return FunctionTransformer(
        func=lambda x: x - 1,
        inverse_func=lambda x: x + 1,
    )
