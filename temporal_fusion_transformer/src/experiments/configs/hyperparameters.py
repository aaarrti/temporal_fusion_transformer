from __future__ import annotations

from typing import Literal

from ml_collections import ConfigDict

Choice = Literal["electricity", "favorita", "hamburg_air_quality"]


def get_config(choice: Choice = "electricity") -> ConfigDict:
    config = ConfigDict()
    config.prng_seed = 69
    config.shuffle_buffer_size = 1024
    config.optimizer = _get_optimizer_config(choice)
    config.model = _get_model_config(choice)
    return config


def _get_model_config(choice: Choice) -> ConfigDict:
    config = {
        "electricity": {
            "num_attention_heads": 8,
            "num_decoder_blocks": 4,
            "latent_dim": 256,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "favorita": {},
        "hamburg_air_quality": {},
    }
    return config[choice]


def _get_optimizer_config(choice: Choice):
    config = {
        "electricity": {
            "init_lr": 1e-4,
            "decay_steps": 0.8,
            "alpha": 0.8,
            "mechanize": False,
            "clipnorm": 0.0,
            "ema": 0.99,
        },
        "favorita": {},
        "hamburg_air_quality": {},
    }
    return config[choice]
