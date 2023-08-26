import jax
from importlib import util

jax.config.update("jax_softmax_custom_jvp", True)
from temporal_fusion_transformer.src.config_dict import ConfigDictProto
from temporal_fusion_transformer.src import training_scripts
from temporal_fusion_transformer.config import get_config
from temporal_fusion_transformer.src import datasets

if util.find_spec("optuna") is not None:
    from temporal_fusion_transformer.src import hyperparams
