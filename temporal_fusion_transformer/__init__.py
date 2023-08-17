from importlib import util


from temporal_fusion_transformer.src.config_dict import ConfigDict

if util.find_spec("polars") is not None:
    # TODO: more granular check
    from temporal_fusion_transformer.src import dataset_scripts

if util.find_spec("absl_extra") is not None:
    # TODO: more granular check
    from temporal_fusion_transformer.src import training_scripts

if util.find_spec("ml_collections"):
    from temporal_fusion_transformer.config import get_config
