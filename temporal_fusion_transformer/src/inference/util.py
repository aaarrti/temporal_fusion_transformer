from __future__ import annotations

from typing import TYPE_CHECKING, Callable, overload

import numpy as np

from temporal_fusion_transformer.src.config_dict import DataConfig, ModelConfig
from temporal_fusion_transformer.src.modeling.tft_model import (
    TemporalFusionTransformer,
    TftOutputs,
)

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.lib_types import PredictFn


@overload
def reload_model(
    filename: str,
    model_config: ModelConfig,
    data_config: DataConfig,
    jit_module: bool,
    return_attention: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    ...


@overload
def reload_model(
    filename: str,
    model_config: ModelConfig,
    data_config: DataConfig,
    jit_module: bool,
    return_attention: bool = True,
) -> Callable[[np.ndarray], TftOutputs]:
    ...


def reload_model(
    filename: str,
    model_config: ModelConfig,
    data_config: DataConfig,
    jit_module: bool = True,
    return_attention: bool = False,
) -> PredictFn:
    from absl_extra.flax_utils import load_from_msgpack

    params = load_from_msgpack(None, filename)
    model = TemporalFusionTransformer.from_config_dict(
        model_config, data_config, jit_module=jit_module, return_attention=return_attention
    )
    return model.bind({"params": params})
