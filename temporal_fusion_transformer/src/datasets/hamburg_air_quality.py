import polars as pl

from src.datasets.base import Triple
from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, DF


class HamburgAirQuality(MultiHorizonTimeSeriesDataset):

    """
    References
    ---------

    - https://repos.hcu-hamburg.de/handle/hcu/893
    """

    def download_data(self, path: str):
        super().download_data(path)

    def read_csv(self, path: str) -> DF:
        super().read_csv(path)

    def split_data(self, df: DF) -> Triple[DF]:
        super().split_data(df)

    def needs_download(self, path: str) -> bool:
        super().needs_download(path)
