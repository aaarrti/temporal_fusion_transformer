from keras.utils.tf_utils import set_random_seed
from tests.constants import PRNG_SEED
from absl_extra import setup_logging


set_random_seed(PRNG_SEED)
setup_logging()
