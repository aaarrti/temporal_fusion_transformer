import os
from importlib import util
from typing import TYPE_CHECKING

from temporal_fusion_transformer.src import datasets, utils
from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.datasets import MultiHorizonTimeSeriesDataset
from temporal_fusion_transformer.src.modeling.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.train_lib import (
    load_dataset_from_config,
    train_model_from_config,
)
