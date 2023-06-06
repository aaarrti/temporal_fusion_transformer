import logging
import tensorflow as tf
import absl.logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
)
tf.get_logger().setLevel("DEBUG")
absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)
