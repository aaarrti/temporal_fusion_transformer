from typing import Literal
from ml_collections import ConfigDict


Choice = Literal["electricity", "favorita", "hamburg_air_quality"]


def get_config(dataset: Choice) -> ConfigDict:
    # fmt: off
    choices = {
        "electricity": {
            "num_encoder_steps": 168,          # 7 * 24
            "total_time_steps": 192,           # 8 * 24
            "num_outputs": 1,
            "known_categories_sizes": [
                12,                            # month
                31,                            # day
                24,                            # hour
                7                              # day of week
            ],
            "static_categories_sizes": [370],  # id
            "input_observed_idx": [],
            "input_static_idx": [5],           # id
            "input_known_real_idx": [0],       # year
            "input_known_categorical_idx": [
                1,                             # month
                2,                             # day
                3,                             # hour
                4                              # day of week
            ],
            
        },
        "favorita": {
            "total_time_steps": 120,
            "num_encoder_steps": 90,
            "num_outputs": 1,
            "known_categories_sizes": [
                12,  # month
                31,  # day of month
                7,   # day of week
                39,  # national holiday
                2,   # regional hol
                6,   # local holiday
                2,   # on promotion
                2,   # open
            ],
            "static_categories_sizes": [
                3586,  # item nbr
                53,    # store nbr
                22,    # city
                16,    # state,
                5,     # type
                17,    # cluster
                32,    # family
                317,   # class
                2,     # perishable
            ],
            "input_observed_idx": [17, 18],
            "input_static_idx": [0, 1, 2, 3, 4, 5, 6, 7],
            "input_known_real_idx": [8],
            "input_known_categorical_idx": [9, 10, 11, 12, 13, 14, 15, 16],
        
        },
        "hamburg_air_quality": {},
    }
    # fmt: on
    return ConfigDict(choices[dataset])
