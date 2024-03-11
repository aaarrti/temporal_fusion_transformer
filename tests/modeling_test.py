import jax.random
import pytest

from temporal_fusion_transformer.modeling.layers import InputEmbedding, EmbeddingStruct
from temporal_fusion_transformer.modeling.model import TemporalFusionTransformer, TftOutputs

key = jax.random.PRNGKey(42)


@pytest.fixture(scope="function")
def embeding():
    return InputEmbedding(
        input_static_idx=[0, 1],
        input_known_real_idx=[2],
        input_known_categorical_idx=[3, 4],
        input_observed_idx=[5],
        latent_dim=8,
        static_categories_sizes=[72, 15],
        known_categories_sizes=[12, 101],
    )


def test_embedding(embeding):
    x = jax.random.uniform(key, shape=(32, 3, 6))
    y: EmbeddingStruct = embeding.init_with_output(key, x)[0]
    assert y.static.shape == (32, 2, 8)
    assert y.observed.shape == (32, 3, 8, 1)
    assert y.known.shape == (32, 3, 8, 3)


def test_model(embeding):
    model = TemporalFusionTransformer(
        total_time_steps=3,
        num_encoder_steps=2,
        num_decoder_blocks=1,
        num_attention_heads=4,
        latent_dim=8,
        embedding_layer=embeding,
        num_non_static_inputs=4,
        num_known_inputs=3,
        num_static_inputs=2,
    )
    x = jax.random.uniform(key, shape=(32, 3, 6))
    y: TftOutputs = model.init_with_output(key, x)[0]
    assert y.logits.shape == (32, 1, 1, 3)
