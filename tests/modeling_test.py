import chex
import jax.numpy as jnp

from temporal_fusion_transformer.src.modeling import TemporalFusionTransformer


def test_dummy_model():
    model = TemporalFusionTransformer(
        input_observed_idx=[],
        input_static_idx=[0],
        input_known_real_idx=[3],
        input_known_categorical_idx=[1, 2],
        static_categories_sizes=[2],
        known_categories_sizes=[2, 2],
        latent_dim=5,
        dropout_rate=0.1,
        num_encoder_steps=20,
        total_time_steps=30,
        num_attention_heads=1,
        num_decoder_blocks=5,
        num_quantiles=3,
    )

    x = jnp.ones(shape=[8, 30, 4], dtype=jnp.float32)
    y = model(x)

    assert y.dtype == jnp.float32
    chex.assert_shape(y, (8, 10, 1, 3))
    chex.assert_tree_all_finite(y)
