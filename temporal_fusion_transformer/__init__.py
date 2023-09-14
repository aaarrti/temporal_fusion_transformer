from typing import TYPE_CHECKING

import jax

jax.config.update("jax_softmax_custom_jvp", True)


if TYPE_CHECKING:
    from temporal_fusion_transformer.src.config_dict import ConfigDict


from temporal_fusion_transformer.src import experiments
from temporal_fusion_transformer.src import util
