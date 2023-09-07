from importlib import util

import jax

jax.config.update("jax_softmax_custom_jvp", True)
from temporal_fusion_transformer.src import datasets, training_scripts
from temporal_fusion_transformer.src.config_dict import ConfigDictProto
from temporal_fusion_transformer.src.tft_model import TemporalFusionTransformer

if util.find_spec("ml_collections") is not None:
    from temporal_fusion_transformer.config import get_config

if util.find_spec("optuna") is not None:
    from temporal_fusion_transformer.src import hyperparams
