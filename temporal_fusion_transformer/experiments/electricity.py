# -------------------------------------------
# Columns are:
#
# Unnamed: 0                   int64
# power_usage                float64
# t                          float64
# days_from_start              int64
# categorical_id              object
# date                        object
# id                          object
# hour                         int64
# day                          int64
# day_of_week                  int64
# month                        int64
# hours_from_start           float64
# categorical_day_of_week      int64
# categorical_hour             int64
# ---------------------------------------------
# Data before pre-processing
#
# 0	17544 2.53807106598985 26304.0 1096	MT_001 2014-01-01 00:00:00 MT_001 0	1 2	1 26304.0 2	0
# 1	17545 2.85532994923858 26305.0 1096	MT_001 2014-01-01 01:00:00 MT_001 1	1 2	1 26305.0 2	1
# 2	17546 2.85532994923858 26306.0 1096	MT_001 2014-01-01 02:00:00 MT_001 2	1 2	1 26306.0 2	2
# 3	17547 2.85532994923858 26307.0 1096	MT_001 2014-01-01 03:00:00 MT_001 3	1 2	1 26307.0 2	3
# 4	17548 2.53807106598985 26308.0 1096	MT_001 2014-01-01 04:00:00 MT_001 4	1 2	1 26308.0 2	4
# ---------------------------------------------
# Data after pre-processing:
#
# 0	17544 -0.1271744626616888 26304.0 1096 0 2014-01-01 00:00:00 MT_001	-1.661324772583615 1 -0.4997188006494831 1 -1.7317213010867616 2 0
# 1	17545 -0.0507126717957727 26305.0 1096 0 2014-01-01 01:00:00 MT_001	-1.516861748880692 1 -0.4997188006494831 1 -1.7310622254250732 2 1
# 2	17546 -0.0507126717957727 26306.0 1096 0 2014-01-01 02:00:00 MT_001	-1.372398725177769 1 -0.4997188006494831 1 -1.7304031497633845 2 2
# 3	17547 -0.0507126717957727 26307.0 1096 0 2014-01-01 03:00:00 MT_001	-1.227935701474846 1 -0.4997188006494831 1 -1.7297440741016958 2 3
# 4	17548 -0.1271744626616888 26308.0 1096 0 2014-01-01 04:00:00 MT_001	-1.083472677771922 1 -0.4997188006494831 1 -1.7290849984400074 2 4
# ----------------------------------------------
# Data batch looks like:
# {
#   'inputs': {... shape=(100, 192, 1), dtype=float} <==> x,
#   'outputs': {... shape=(100, 24, 1), dtype=float} <==> y,
#   'active_entries': {... shape=(100, 24, 1), dtype=float} <==> sample weights
#   'time': {... shape=(100, 192, 1), dtype=float},
#   'identifier': {..., shape=(100, 192, 1), dtype=string}
# }
# where (I guess):
# - 100 is batch size
# - 192 is number of past observations
# - 24 is number of future observations
# ------------------------------------------------------------
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Dict, Type

from temporal_fusion_transformer.data_formatters.electricity import ElectricityFormatter
from temporal_fusion_transformer.utils import classproperty
from temporal_fusion_transformer.experiments.experiment import Experiment


class ElectricityExperiment(Experiment):
    @classproperty
    def data_formatter(self) -> Type[ElectricityFormatter]:
        raise ElectricityFormatter

    @classproperty
    def time_steps_config(self) -> Dict[str, int]:
        return {
            "total_time_steps": 8 * 24,
            "num_encoder_steps": 7 * 24,
        }

    @classproperty
    def default_model_params(self) -> Dict[str, ...]:
        return {
            "dropout_rate": 0.1,
            "hidden_layer_size": 160,
            "learning_rate": 0.001,
            "minibatch_size": 64,
            "max_gradient_norm": 0.01,
            "num_heads": 4,
            "stack_size": 1,
        }
