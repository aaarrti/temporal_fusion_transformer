import logging
import tensorflow as tf
import absl.logging
from keras.utils.tf_utils import set_random_seed

set_random_seed(42)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
)
tf.get_logger().setLevel("DEBUG")
absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)
