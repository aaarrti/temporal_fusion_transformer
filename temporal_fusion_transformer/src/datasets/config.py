from typing import Literal


def get_config(dataset: Literal["electricity", "favorita"]):
    # fmt: off
    choices = {
        "electricity": {
            "num_encoder_steps": 168,          # 7 * 24
            "total_time_steps": 192,           # 8 * 24
            "num_outputs": 1,
            "known_categories_sizes": [
                12,                            # month
                21,                            # day
                24,                            # hour
                7                              # day of week
            ],
            "static_categories_sizes": [369],  # id
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
        
        }
    }
    # fmt: on
    return choices[dataset]
