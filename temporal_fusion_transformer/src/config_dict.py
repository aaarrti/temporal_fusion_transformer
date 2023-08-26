from __future__ import annotations

from typing import Protocol, Sequence


class ConfigDictProto(Protocol):
    prng_seed: int
    shuffle_buffer_size: int
    fixed_params: FixedParamsConfig
    hyperparams: HyperParamsConfig
    optimizer: OptimizerConfig


class HyperParamsConfig(Protocol):
    """
    Attributes
    ----------

    num_attention_heads:
    num_decoder_blocks:
    dropout_rate:
    latent_dim:
        Model latent space dimensionality.

    quantiles:
        Quantiles use for loss function.
    """

    num_attention_heads: int
    num_decoder_blocks: int
    latent_dim: int
    dropout_rate: int
    quantiles: Sequence[float]


class FixedParamsConfig(Protocol):
    """
    Attributes
    ----------

    num_encoder_steps:
        Number of time-steps considered past.
    total_time_steps:
        Total number observed time-steps.
    num_outputs:
        Number of values to predict.
    known_categories_sizes:
        Highest possible values for known categorical inputs (in order).
    static_categories_sizes:
         Highest possible values for static inputs (in order).

    input_observed_idx:
        Indices, at which observed inputs are in stacked array.
    input_static_idx:
        Indices, at which static inputs are in stacked array.
    input_known_real_idx:
        Indices, at which known real inputs are in stacked array.
    input_known_categorical_idx:
        Indices, at which known categorical inputs are in stacked array.

    """

    num_encoder_steps: int
    total_time_steps: int
    num_outputs: int
    known_categories_sizes: Sequence[int]
    static_categories_sizes: Sequence[int]
    input_observed_idx: Sequence[int]
    input_static_idx: Sequence[int]
    input_known_real_idx: Sequence[int]
    input_known_categorical_idx: Sequence[int]


class OptimizerConfig(Protocol):
    """
    Attributes
    ----------

    learning_rate:
        Initial learning rate
    decay_steps:
        Fraction of total training steps over which to decay. If set to 0, constant learning_rate will be used.
    decay_alpha:
        Fraction of initial learning rate, which will be reached after `decay_steps`
    ema:
        Only applies if `use_ema==True`


    """

    learning_rate: float
    decay_steps: float
    decay_alpha: float
    ema: float
    clipnorm: float
