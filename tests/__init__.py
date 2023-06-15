from keras.utils.tf_utils import set_random_seed
from temporal_fusion_transformer import setup_logging
from tests.constants import PRNG_SEED


set_random_seed(PRNG_SEED)
setup_logging()
