from importlib import util

if util.find_spec("polars") is not None:
    from temporal_fusion_transformer.src.datasets.base import MultiHorizonTimeSeriesDataset, downcast_dataframe
    from temporal_fusion_transformer.src.datasets.electricity import Electricity
    from temporal_fusion_transformer.src.datasets.favorita import Favorita
