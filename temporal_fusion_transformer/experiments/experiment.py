from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, NamedTuple, List

from temporal_fusion_transformer.data_formatters.data_formatter import DataFormatter
from temporal_fusion_transformer.utils import classproperty


class ExperimentConfig(NamedTuple):
    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int


class Experiment(ABC):
    @classproperty
    @abstractmethod
    def data_formatter(self) -> Type[DataFormatter]:
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def time_steps_config(self) -> TimeStepsConfig:
        raise NotImplementedError

    @classproperty
    def experiment_params(self) -> Dict[str, ...]:
        """Returns fixed model parameters for experiments."""

        required_keys = [
            "total_time_steps",
            "num_encoder_steps",
            "num_epochs",
            "early_stopping_patience",
            "multiprocessing_workers",
        ]

        fixed_params = self.time_steps_config

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError(
                    "Field {}".format(k) + " missing from fixed parameter definitions!"
                )

        fixed_params[
            "column_definition"
        ] = self.data_formatter.make_column_definitions()
        fixed_params.update(self.data_formatter.get_tft_input_indices())
        return fixed_params

    @classproperty
    @abstractmethod
    def default_model_params(self) -> Dict[str, ...]:
        raise NotImplementedError
