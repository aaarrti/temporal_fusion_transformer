from importlib import util

if util.find_spec("ml_collections") is not None:
    from temporal_fusion_transformer.src.datasets.config import get_config

if util.find_spec("sklearn") is not None:
    from temporal_fusion_transformer.src.datasets.preprocessing import (
        serialize_preprocessor,
    )

if util.find_spec("polars") is not None:
    from temporal_fusion_transformer.src.datasets import electricity, favorita
