import os

from absl_extra.logging_utils import setup_logging

setup_logging()
os.environ["KERAS_BACKEND"] = "jax"

import jax

jax.config.update("jax_include_full_tracebacks_in_locations", True)
jax.config.update("jax_traceback_filtering", "off")

from keras_core.config import disable_traceback_filtering
from keras_core.src.utils.io_utils import set_logging_verbosity

disable_traceback_filtering()
set_logging_verbosity("DEBUG")
