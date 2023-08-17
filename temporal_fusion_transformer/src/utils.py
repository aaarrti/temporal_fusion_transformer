from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import jax.numpy as jnp
from absl import logging

from temporal_fusion_transformer.src.config_dict import FixedParamsConfig

if TYPE_CHECKING:
    from temporal_fusion_transformer.src.tft_layers import ComputeDtype, InputStruct


def make_input_struct_from_config(
    x_batch: jnp.ndarray, config: FixedParamsConfig, dtype: ComputeDtype = jnp.float32
) -> InputStruct:
    return make_input_struct_from_idx(
        x_batch,
        config.input_static_idx,
        config.input_known_real_idx,
        config.input_known_categorical_idx,
        config.input_observed_idx,
        dtype=dtype,
    )


def make_input_struct_from_idx(
    x_batch: jnp.ndarray,
    input_static_idx,
    input_known_real_idx,
    input_known_categorical_idx,
    input_observed_idx,
    dtype: ComputeDtype = jnp.float32,
) -> InputStruct:
    from temporal_fusion_transformer.src.tft_layers import InputStruct

    declared_num_features = (
        len(input_static_idx) + len(input_known_real_idx) + len(input_known_categorical_idx) + len(input_observed_idx)
    )
    num_features = x_batch.shape[-1]

    if num_features != declared_num_features:
        unknown_indexes = sorted(
            list(
                set(
                    input_static_idx + input_known_real_idx + input_known_categorical_idx + input_observed_idx
                ).symmetric_difference(range(num_features))
            )
        )
        if num_features > declared_num_features:
            logging.error(
                f"Declared number of features does not match with the one seen in input, "
                f"could not indentify inputs at {unknown_indexes}"
            )
            unknown_indexes = jnp.asarray(unknown_indexes, jnp.int32)
            unknown_inputs = jnp.take(x_batch, unknown_indexes, axis=-1).astype(dtype)
        else:
            logging.error(
                f"Declared number of features does not match with the one seen in input, "
                f"no inputs at {unknown_indexes}"
            )
            unknown_inputs = None
    else:
        unknown_inputs = None

    static = jnp.take(x_batch, jnp.asarray(input_static_idx), axis=-1).astype(jnp.int32)

    if len(input_known_real_idx) > 0:
        known_real = jnp.take(x_batch, jnp.asarray(input_known_real_idx), axis=-1).astype(dtype)
    else:
        known_real = None

    if len(input_known_categorical_idx) > 0:
        known_categorical = jnp.take(x_batch, jnp.asarray(input_known_categorical_idx), axis=-1).astype(jnp.int32)
    else:
        known_categorical = None

    if len(input_observed_idx) > 0:
        observed = jnp.take(x_batch, jnp.asarray(input_observed_idx), axis=-1).astype(dtype)
    else:
        observed = None

    return InputStruct(
        static=static,
        known_real=known_real,
        known_categorical=known_categorical,
        observed=observed,
        unknown=unknown_inputs,
    )


def make_timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")
