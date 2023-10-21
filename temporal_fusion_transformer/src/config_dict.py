from __future__ import annotations

from typing import Protocol, Sequence, Union
from ml_collections import config_dict


class _DataConfig(Protocol):
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
    """

    encoder_steps: int
    total_time_steps: int
    num_outputs: int
    known_categories_sizes: Sequence[int]
    static_categories_sizes: Sequence[int]
    input_observed_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]
    input_static_idx: Sequence[int]


class _ModelConfig(Protocol):
    """
    Attributes
    ----------
    num_attention_heads:
    num_decoder_blocks:
    hidden_layer_size:
    dropout_rate:
    unroll:
        If set to true, will unroll LSTM cell. Can be beneficial for XLA backend. Will increase memory consumption.
        For pure TensorFlow (Grappler) backend, set to `False` to allow using CuDNN implementation.
    """

    num_attention_heads: int
    num_decoder_blocks: int
    hidden_layer_size: int
    dropout_rate: int
    unroll: bool


class _OptimizerConfig(Protocol):
    """
    Attributes
    ----------
    learning_rate:
        Initial value for (cosine) learning rate schedule.
    decay_steps:
        Fraction of total training steps over which to decay. If set to 0.0, will use constant learning rate.
    alpha:
        Fraction of original learning rate until which to decay. Only valid if `decay_steps` != 0.0
    clipnorm:
        Highest norm, up to which gradients will be clipped. If set to 0, will not clip.
    use_ema:
        Use EMA optimizer.
    weight_decay:
    """

    learning_rate: float
    decay_steps: float
    alpha: float
    clipnorm: float
    use_ema: bool
    weight_decay: float


DataConfig = Union[_DataConfig, config_dict.FrozenConfigDict]
ModelConfig = Union[_ModelConfig, config_dict.FrozenConfigDict]
OptimizerConfig = Union[_OptimizerConfig, config_dict.FrozenConfigDict]


class Config(Protocol):
    """
    Attributes
    ----------
    data:
        Fixed (caused by data) parameters.
    model:
        Hyperparameters we can fine-tune.
    optimizer:
    quantiles:
        Quantiles used for loss calculation.
    """

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    quantiles: Sequence[float]
