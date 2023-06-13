import logging
import tensorflow as tf
import absl.logging
from keras.utils.tf_utils import set_random_seed

from tests.constants import PRNG_SEED


set_random_seed(PRNG_SEED)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
)
tf.get_logger().setLevel("DEBUG")
absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)
