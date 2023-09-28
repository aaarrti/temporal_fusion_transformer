from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Callable, TypedDict

    from jax.random import KeyArray
    import jax.numpy as jnp
    from flax.training.early_stopping import EarlyStopping
    from flax.training.dynamic_scale import DynamicScale
    from absl_extra.flax_utils import TrainingHooks
    from temporal_fusion_transformer.src.modeling.tft_model import TftOutputs
    from temporal_fusion_transformer.src.training.training_hooks import (
        HooksConfig,
        EarlyStoppingConfig,
    )

    class PRNGCollection(TypedDict):
        dropout: KeyArray
        lstm: KeyArray

    ComputeDtype = jnp.float32 | jnp.bfloat16 | jnp.float16
    HooksT = TrainingHooks | Literal["auto"] | HooksConfig | None
    DynamicScaleT = DynamicScale | None | Literal["auto"]
    EarlyStoppingT = EarlyStopping | None | Literal["auto"] | EarlyStoppingConfig
    DeviceTypeT = Literal["gpu", "tpu"]
    LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    PredictFn = Callable[[jnp.ndarray], jnp.ndarray | TftOutputs]
