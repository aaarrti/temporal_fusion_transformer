from importlib import util

from temporal_fusion_transformer.src.datasets.preprocessing import serialize_preprocessor
from temporal_fusion_transformer.src.datasets.config import get_config

if util.find_spec("polars") is not None:
    from temporal_fusion_transformer.src.datasets import electricity
    from temporal_fusion_transformer.src.datasets import favorita
