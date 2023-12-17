import numpy as np

from temporal_fusion_transformer.src.modeling.modeling import TemporalFusionTransformer
from temporal_fusion_transformer.src.modeling.modeling_v2 import (
    TemporalFusionTransformer as TemporalFusionTransformerV2,
)


def test_dummy_model():
    model = TemporalFusionTransformer(
        input_observed_idx=[],
        input_static_idx=[0],
        input_known_real_idx=[3],
        input_known_categorical_idx=[1, 2],
        static_categories_sizes=[2],
        known_categories_sizes=[2, 2],
        hidden_layer_size=5,
        dropout_rate=0.1,
        encoder_steps=20,
        total_time_steps=30,
        num_attention_heads=1,
        num_decoder_blocks=5,
        num_quantiles=3,
    )
    
    x = np.ones(shape=[8, 30, 4], dtype=np.float32)
    y = model(x)
    
    assert y.dtype == np.float32
    assert y.shape == (8, 10, 1, 3)


def test_dummy_model_v2(capsys):
    with capsys.disabled():
        model = TemporalFusionTransformerV2(
            input_observed_idx=[],
            input_static_idx=[0],
            input_known_real_idx=[3],
            input_known_categorical_idx=[1, 2],
            static_categories_sizes=[2],
            known_categories_sizes=[2, 2],
            hidden_layer_size=5,
            dropout_rate=0.1,
            encoder_steps=20,
            total_time_steps=30,
            num_attention_heads=1,
            num_decoder_blocks=5,
            num_quantiles=3,
        )
        
        x = np.ones(shape=[8, 30, 4], dtype=np.float32)
        y = model(x)
        
        assert y.dtype == np.float32
        assert y.shape == (8, 10, 1, 3)
