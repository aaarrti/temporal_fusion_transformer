from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

import tomli

RegularizerT = Optional[Literal["L1", "L2", "L1_L2"]]


@dataclass(frozen=True, repr=False)
class Config:
    """
    Attributes
    ----------
    encoder_steps:
        Number of time steps considered past.
    total_time_steps:
        Number of time steps (in the future), which we try to forecast.
    num_outputs:
    known_categories_sizes:
    static_categories_sizes:
    input_observed_idx:
    input_known_real_idx:
    input_known_categorical_idx:
    input_static_idx:
    num_attention_heads:
    num_decoder_blocks:
    hidden_layer_size:
    dropout_rate:
    learning_rate:
        Initial value for (cosine) learning rate schedule.
    decay_steps:
        Fraction of total training steps over which to decay. If set to 0.0, will use constant learning rate.
    decay_alpha:
        Fraction of original learning rate until which to decay. Only valid if `decay_steps` != 0.0
    clipnorm:
        Highest norm, up to which gradients will be clipped. If set to 0, will not clip.
    use_ema:
        Use EMA optimizer.
    weight_decay:
    """

    # fixed params
    encoder_steps: int
    total_time_steps: int
    known_categories_sizes: Sequence[int]
    static_categories_sizes: Sequence[int]
    input_observed_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]
    input_static_idx: Sequence[int]
    num_attention_heads: int
    num_decoder_blocks: int
    hidden_layer_size: int
    learning_rate: float
    batch_size: int = 64
    epochs: int = 10
    validation_boundary: Any = None
    test_boundary: Any = None
    split_overlap: int = None
    test_split_save_format: Literal["csv", "parquet"] = "parquet"
    dropout_rate: int = 0.1
    compression: Literal["GZIP"] | None = "GZIP"
    drop_remainder: bool = True
    shuffle_buffer_size: int = 1024
    num_outputs: int = 1
    decay_steps: float | None = None
    decay_alpha: float | None = None
    clipnorm: float | None = None
    use_ema: bool = False
    weight_decay: float | None = None
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    kernel_regularizer: RegularizerT = None
    bias_regularizer: RegularizerT = None
    activity_regularizer: RegularizerT = None
    recurrent_regularizer: RegularizerT = None

    def __str__(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=4, sort_keys=True, default=str)

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self), indent=4, sort_keys=True, default=str)

    @staticmethod
    def read_from_file(path: str) -> Config:
        with open(path, "rb") as file:
            content = tomli.load(file)

        content_copy = content.copy()

        for k, v in content.items():
            if v == "None":
                content_copy[k] = None

        return Config(**content_copy)
