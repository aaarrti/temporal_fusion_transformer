from __future__ import annotations

from typing import Literal

from ml_collections import ConfigDict

Choice = Literal["electricity", "favorita", "hamburg_air_quality"]


def get_config(choice: Choice = "electricity") -> ConfigDict:
    config = ConfigDict()
    config.prng_seed = 69
    config.shuffle_buffer_size = 2048
    config.optimizer = get_optimizer_config(choice)
    config.model = get_model_config(choice)
    return config


def get_model_config(choice: Choice) -> ConfigDict:
    config = {
        "electricity": {
            "num_attention_heads": 10,
            "num_decoder_blocks": 5,
            "latent_dim": 160,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "favorita": {
            "dropout_rate": 0.1,
            "latent_dim": 240,
            "num_attention_heads": 4,
            "num_decoder_blocks": 1,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "hamburg_air_quality": {},
    }
    return config[choice]


def get_optimizer_config(choice: Choice):
    config = {
        "electricity": {
            "clipnorm": 0.0,
            "decay_alpha": 0.1,
            "decay_steps": 0.8,
            "ema": 0.99,
            "learning_rate": 5e-4,
        },
        "favorita": {
            "learning_rate": 1e-3,
            "clipnorm": 100.0,
            "decay_alpha": 0.1,
            "decay_steps": 0.8,
            "ema": 0.99,
        },
        "hamburg_air_quality": {},
    }
    return config[choice]
