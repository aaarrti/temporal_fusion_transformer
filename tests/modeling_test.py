import jax.random
import jax.numpy as jnp

from temporal_fusion_transformer.src.modeling.loss_fn import quantile_pinball_loss, pinball_loss
from temporal_fusion_transformer.src.modeling.model import TemporalFusionTransformer
from temporal_fusion_transformer.src.modeling.layers import InputEmbedding, EmbeddingStruct


key = jax.random.PRNGKey(42)


def test_embedding():
    layer = InputEmbedding(
        input_static_idx=[0, 1],
        input_known_real_idx=[2],
        input_known_categorical_idx=[3, 4],
        input_observed_idx=[5],
        latent_dim=8,
        static_categories_sizes=[72, 15],
        known_categories_sizes=[12, 101],
    )
    x = jax.random.uniform(key, shape=(32, 3, 6))
    y: EmbeddingStruct = layer.init_with_output(key, x)[0]

    assert y.static.shape == (32, 2, 8)
    assert y.observed.shape == (32, 3, 8, 1)
    assert y.known.shape == (32, 3, 8, 3)


def test_model():
    model = TemporalFusionTransformer(
        total_time_steps=3,
        num_encoder_steps=2,
        num_decoder_blocks=1,
        num_attention_heads=4,
        latent_dim=8,
        input_static_idx=[0, 1],
        input_known_real_idx=[2],
        input_known_categorical_idx=[3, 4],
        input_observed_idx=[5],
        static_categories_sizes=[72, 15],
        known_categories_sizes=[12, 101],
    )
    x = jax.random.uniform(key, shape=(32, 3, 6))
    y = model.init_with_output(key, x)[0].logits
    assert y.shape == (32, 1, 1, 3)


def test_loss_fn():
    y_true = jnp.ones((8, 2, 1), dtype=jnp.float32)
    y_pred = jnp.ones((8, 2, 1), dtype=jnp.float32)
    loss = pinball_loss(y_true, y_pred, 0.5)
    assert loss.shape == (8,)


def test_quantile_loss_fn():
    y_true = jnp.ones((8, 2, 1), dtype=jnp.float32)
    y_pred = jnp.ones((8, 2, 1, 3), dtype=jnp.float32)
    loss = quantile_pinball_loss(y_true, y_pred)
    assert loss.shape == (8,)
