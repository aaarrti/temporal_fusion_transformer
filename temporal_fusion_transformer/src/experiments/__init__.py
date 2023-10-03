#from importlib import util

from temporal_fusion_transformer.src.experiments import util

# if util.find_spec("ml_collections") is not None:
from temporal_fusion_transformer.src.experiments.configs import (
    fixed_parameters,
    hyperparameters,
)

# if util.find_spec("polars") is not None and util.find_spec("tensorflow") is not None:
from temporal_fusion_transformer.src.experiments.electricity import Electricity
from temporal_fusion_transformer.src.experiments.favorita import Favorita
