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


from temporal_fusion_transformer.utils import make_tft_model, make_gpu_strategy
from temporal_fusion_transformer.plotting import plot_predictions
