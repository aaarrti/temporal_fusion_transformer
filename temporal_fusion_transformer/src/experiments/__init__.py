from importlib import util

if util.find_spec("polars") is not None and util.find_spec("tensorflow") is not None:
    from temporal_fusion_transformer.src.experiments.base import (
        Experiment,
        MultiHorizonTimeSeriesDataset,
        Preprocessor,
    )
    from temporal_fusion_transformer.src.experiments.electricity import Electricity

if util.find_spec("ml_collection") is not None:
    from temporal_fusion_transformer.src.experiments.config import get_config
