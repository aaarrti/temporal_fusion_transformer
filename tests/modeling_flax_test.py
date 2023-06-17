import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from temporal_fusion_transformer.src.modeling_flax import (
    TemporalFusionTransformer,
    StaticCovariatesEncoder,
    TFTInputEmbedding,
    VariableSelection,
    EncoderBlock,
    TFTInput,
    ContextInput,
)
from tests.constants import PRNG_SEED
from tests import lifted_variants

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5
prng_key = jax.random.PRNGKey(PRNG_SEED)


class TFTLayersTest(chex.TestCase, parameterized.TestCase):
    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    def test_input_embedding(self):
        x_batch = make_x_batch()
        layer = self.variant(TFTInputEmbedding)(
            static_categories_sizes,
            known_categories_sizes,
            hidden_layer_size,
        )
        x_embeds, _ = layer.init_with_output(prng_key, x_batch)
        # Same shapes as in the original implementation.
        chex.assert_shape(x_embeds.static, (batch_size, 2, hidden_layer_size))
        chex.assert_shape(
            x_embeds.known,
            (batch_size, n_time_steps, hidden_layer_size, 4),
        )
        chex.assert_shape(
            x_embeds.observed,
            (batch_size, n_time_steps, hidden_layer_size, 1),
        )

    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    def test_static_covariates_encoder(self):
        layer = self.variant(StaticCovariatesEncoder)(
            hidden_layer_size,
            dropout_rate=0.0,
        )
        static_context, _ = layer.init_with_output(
            prng_key, jax.random.uniform(prng_key, (batch_size, 2, hidden_layer_size))
        )
        # Same shapes as in the original implementation.
        chex.assert_shape(
            static_context.vector,
            (batch_size, hidden_layer_size),
        )
        chex.assert_shape(
            static_context.enrichment,
            (batch_size, hidden_layer_size),
        )
        chex.assert_shape(
            static_context.state_h,
            (batch_size, hidden_layer_size),
        )
        chex.assert_shape(
            static_context.state_c,
            (batch_size, hidden_layer_size),
        )

    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    @parameterized.parameters((25, 4), (5, 3))
    def test_temporal_variable_selection_network(self, time_steps, features):
        layer = self.variant(VariableSelection)(hidden_layer_size, dropout_rate=0)
        x = ContextInput(
            inputs=jax.random.uniform(
                prng_key, (batch_size, time_steps, hidden_layer_size, features)
            ),
            context=jax.random.uniform(prng_key, (batch_size, hidden_layer_size)),
        )
        (feature, flags, _), _ = layer.init_with_output(prng_key, x)
        # Same shapes as in the original implementation.
        chex.assert_shape(feature, (batch_size, time_steps, hidden_layer_size))
        chex.assert_shape(flags, (batch_size, time_steps, 1, features))

    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    def test_decoder(self):
        layer = self.variant(EncoderBlock)(
            num_attention_heads=hidden_layer_size,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=0,
        )
        decoder_out, _ = layer.init_with_output(
            prng_key,
            jax.random.uniform(
                prng_key,
                (batch_size, n_time_steps, hidden_layer_size),
            ),
        )
        chex.assert_shape(
            decoder_out,
            (batch_size, n_time_steps, hidden_layer_size),
        )


class TFTModelTest(chex.TestCase, parameterized.TestCase):
    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    @parameterized.parameters(1, 4, 12)
    def test_tft_model(self, num_stacks):
        x_batch = make_x_batch()
        model = self.variant(TemporalFusionTransformer)(
            num_encoder_steps=25,
            num_attention_heads=hidden_layer_size,
            hidden_layer_size=hidden_layer_size,
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=known_categories_sizes,
            dropout_rate=0,
            quantiles=[1, 2, 3],
            output_size=1,
            num_stacks=num_stacks,
        )
        logits, _ = model.init_with_output(prng_key, x_batch)
        chex.assert_tree_all_finite(logits, "Test Failed.")
        # 3 is default number of quantiles.
        chex.assert_shape(logits, (batch_size, 5, 3))

    @lifted_variants.variants(
        with_lifted_jit=True,
        without_lifted_jit=True,
    )
    @parameterized.named_parameters(
        (
            "known_categorical==None",
            TFTInput(
                static=jnp.ones((8, 4), dtype=jnp.int32),
                known_real=jnp.ones((8, 30, 3), dtype=jnp.float32),
                observed=jax.random.uniform(prng_key, (8, 30, 1), dtype=jnp.float32),
            ),
            [],
        ),
        (
            "observed==None",
            TFTInput(
                static=jnp.ones((8, 4), dtype=jnp.int32),
                known_real=jnp.ones((8, 30, 3), dtype=jnp.float32),
                known_categorical=jnp.ones((8, 30, 2), dtype=jnp.int32),
            ),
            known_categories_sizes,
        ),
        (
            "known_categorical==None, observed==None",
            TFTInput(
                static=jnp.ones((8, 4), dtype=jnp.int32),
                known_real=jnp.ones((8, 30, 3), dtype=jnp.float32),
                observed=jax.random.uniform(prng_key, (8, 30, 1), dtype=jnp.float32),
            ),
            [],
        ),
    )
    def test_input_is_missing(self, x_batch, _known_categories_sizes):
        model = self.variant(TemporalFusionTransformer)(
            num_encoder_steps=25,
            num_attention_heads=4,
            hidden_layer_size=hidden_layer_size,
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=_known_categories_sizes,
            dropout_rate=0,
            output_size=1,
        )
        logits, _ = model.init_with_output(prng_key, x_batch)
        chex.assert_tree_all_finite(logits, "Test Failed.")
        # 3 is default number of quantiles.
        chex.assert_shape(logits, (batch_size, 5, 3))


def make_x_batch():
    return TFTInput(
        static=jnp.ones((8, 4), dtype=jnp.int32),
        known_real=jnp.ones((8, 30, 3), dtype=jnp.float32),
        known_categorical=jnp.ones((8, 30, 2), dtype=jnp.int32),
        observed=jax.random.uniform(prng_key, (8, 30, 1), dtype=jnp.float32),
    )
