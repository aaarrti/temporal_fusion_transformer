from importlib import util

from temporal_fusion_transformer.src.datasets import preprocessing

if util.find_spec("polars") is not None:
    from temporal_fusion_transformer.src.datasets import electricity
