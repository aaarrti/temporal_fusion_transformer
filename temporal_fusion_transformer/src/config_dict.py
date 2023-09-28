from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, Union

import ml_collections

if TYPE_CHECKING:

    class _ConfigDictProto(Protocol):
        prng_seed: int
        shuffle_buffer_size: int
        model: ModelConfig
        optimizer: OptimizerConfig

    class _ModelConfig(Protocol):
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
        attention_dropout_rate: float

    class _DataConfig(Protocol):
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

    class _OptimizerConfig(Protocol):
        """
        Attributes
        ----------
        ema:
            EMA multiplier, if set to 0 will use gradients directly.
        clipnorm:
            The maximum allowed ratio of update norm to parameter norm.
        init_lr:
            Initial learning rate.
        decay_steps:
            % of training steps, after which `end_lr` must be reached.

        """

        init_lr: float
        decay_steps: float
        alpha: float
        ema: float
        clipnorm: float

    ConfigDict = Union[ml_collections.ConfigDict, _ConfigDictProto]
    OptimizerConfig = Union[ml_collections.ConfigDict, _OptimizerConfig]
    ModelConfig = Union[ml_collections.ConfigDict, _ModelConfig]
    DataConfig = Union[ml_collections.ConfigDict, _DataConfig]
