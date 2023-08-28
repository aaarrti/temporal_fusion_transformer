from __future__ import annotations

from typing import Literal
from ml_collections import ConfigDict


Choice = Literal["electricity", "favorita", "hamburg_air_quality"]


def get_config(choice: Choice = "electricity") -> ConfigDict:
    config = ConfigDict()
    config.prng_seed = 69
    config.shuffle_buffer_size = 2048
    config.optimizer = get_optimizer_config(choice)
    config.hyperparams = get_hyperparams_config(choice)
    config.fixed_params = get_fixed_params_config(choice)
    return config


def get_hyperparams_config(choice: Choice = "electricity") -> ConfigDict:
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


def get_fixed_params_config(choice: Choice = "electricity") -> ConfigDict:
    config = {
        "electricity": {
            "num_encoder_steps": 7 * 24,
            "total_time_steps": 8 * 24,
            "num_outputs": 1,
            "known_categories_sizes": [31, 24, 7, 12],  # day of month  # hour of the day  # day of the week  # month
            "static_categories_sizes": [369],  # id
            "input_observed_idx": [],
            "input_static_idx": [0],
            "input_known_real_idx": [1, 2],
            "input_known_categorical_idx": [3, 4, 5, 6],
        },
        "favorita": {
            "total_time_steps": 120,
            "num_encoder_steps": 90,
            "num_outputs": 1,
            "known_categories_sizes": [
                12,  # month
                31,  # day of month
                7,  # day of week
                39,  # national holiday
                2,  # regional hol
                6,  # local holiday
                2,  # on promotion
                2,  # open
            ],
            "static_categories_sizes": [
                3586,  # item nbr
                53,  # store nbr
                22,  # city
                16,  # state,
                5,  # type
                17,  # cluster
                32,  # family
                317,  # class
                2,  # perishable
            ],
            "input_observed_idx": [17, 18],
            "input_static_idx": [0, 1, 2, 3, 4, 5, 6, 7],
            "input_known_real_idx": [8],
            "input_known_categorical_idx": [9, 10, 11, 12, 13, 14, 15, 16],
        },
        "hamburg_air_quality": {},
    }
    return config[choice]


def get_optimizer_config(choice: Choice = "electricity"):
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
