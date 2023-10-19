import os
from absl_extra.logging_utils import setup_logging

setup_logging()
os.environ["KERAS_BACKEND"] = "jax"
