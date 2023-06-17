def setup_logging():
    import logging
    import tensorflow as tf
    import absl.logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    )
    tf.get_logger().setLevel("DEBUG")
    absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)


from importlib import util
from temporal_fusion_transformer.src.utils import make_tft_model
from temporal_fusion_transformer.src.plotting import plot_predictions
from temporal_fusion_transformer.src import experiments

if util.find_spec("flax") is not None:
    from temporal_fusion_transformer.src.utils import make_flax_tft_model
    from temporal_fusion_transformer.src import training_flax
