import numpy as np

from temporal_fusion_transformer.src.config import Config
from temporal_fusion_transformer.src.modeling.tft_model import TemporalFusionTransformer
from temporal_fusion_transformer.src.utils import count_inputs

# import tensorflow as tf
# tf.config.run_functions_eagerly(True)


def test_air_passengers_model():
    config = Config.read_from_file("temporal_fusion_transformer/configs/air_passengers.toml")
    model = TemporalFusionTransformer.from_dataclass_config(config)
    x = np.ones(
        shape=(config.batch_size, config.total_time_steps, count_inputs(config)), dtype=float
    )
    y = model.predict(x)
    assert y.shape == (
        config.batch_size,
        config.total_time_steps - config.encoder_steps,
        config.num_outputs,
        len(config.quantiles),
    )


def test_electricity_model():
    config = Config.read_from_file("temporal_fusion_transformer/configs/electricity.toml")
    model = TemporalFusionTransformer.from_dataclass_config(config)
    x = np.ones(
        shape=(config.batch_size, config.total_time_steps, count_inputs(config)), dtype=float
    )
    y = model.predict(x)
    assert y.shape == (
        config.batch_size,
        config.total_time_steps - config.encoder_steps,
        config.num_outputs,
        len(config.quantiles),
    )


def test_cari_model():
    config = Config.read_from_file("tests/configs/cari.toml")
    model = TemporalFusionTransformer.from_dataclass_config(config)
    x = np.ones(
        shape=(config.batch_size, config.total_time_steps, count_inputs(config)), dtype=float
    )
    y = model.predict(x)
    assert y.shape == (
        config.batch_size,
        config.total_time_steps - config.encoder_steps,
        config.num_outputs,
        len(config.quantiles),
    )
