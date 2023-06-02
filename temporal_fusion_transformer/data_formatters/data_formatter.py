# coding=utf-8
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

"""
Default data formatting functions for experiments.
For new datasets, inherit form GenericDataFormatter and implement
all abstract functions.

These dataset-specific methods:
1) Define the column and input types for tabular dataframes used by model
2) Perform the necessary input feature engineering & normalisation steps
3) Reverts the normalisation for predictions
4) Are responsible for train, validation and test splits
"""

from __future__ import annotations
import abc
from enum import auto, IntEnum
from abc import abstractmethod, ABC
from typing import List, Tuple, TYPE_CHECKING

import pandas as pd

from temporal_fusion_transformer.utils import classproperty

if TYPE_CHECKING:
    from temporal_fusion_transformer.modeling import TFTInputs


class DataType(IntEnum):
    REAL_VALUED = auto()
    CATEGORICAL = auto()
    DATE = auto()


class InputType(IntEnum):
    TARGET = auto()
    OBSERVED_INPUT = auto()
    KNOWN_INPUT = auto()
    STATIC_INPUT = auto()
    # Single column used as an entity identifier
    ID = auto()
    # Single column exclusively used as a time index
    TIME = auto()


class DataFormatter(ABC):
    @classproperty
    @abstractmethod
    def column_definition(self) -> List[Tuple[str, DataType, InputType]]:
        # TODO replace with named tuple once I figure out what the actual fuck is going on here
        raise NotImplemented

    @classproperty
    @abstractmethod
    def num_classes_per_categorical_input(self) -> List[int]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dataframe(cls, df: pd.DataFrame) -> DataFormatter:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_split(self) -> TFTInputs:
        raise NotImplementedError

    @property
    @abstractmethod
    def validation_split(self) -> TFTInputs:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_split(self) -> TFTInputs:
        raise NotImplementedError

    # @abstractmethod
    # def transform_inputs(self, df):
    #    """Performs feature transformation."""
    #    raise NotImplementedError()
    #
    # @abstractmethod
    # def format_predictions(self, df):
    #    """Reverts any normalisation to give predictions in original scale."""
    #    raise NotImplementedError()
