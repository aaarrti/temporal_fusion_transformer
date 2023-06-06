import tensorflow as tf
from temporal_fusion_transformer.modeling import (
    TemporalFusionTransformer,
    TFTInputs,
    StaticCovariatesEncoder,
    TFTInputEmbedding,
    TemporalVariableSelectionNetwork,
    ContextAwareInputs,
    TemporalFusionDecoder,
    DecoderInputs,
)
from absl.testing import parameterized

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5
PRNG_SEED = 42
# tf.config.run_functions_eagerly(True)


class TFTLayersTest(tf.test.TestCase, parameterized.TestCase):
    def test_input_embedding(self):
        x_batch = make_x_batch()
        layer = TFTInputEmbedding(
            static_categories_sizes,
            known_categories_sizes,
            hidden_layer_size,
        )
        x_embeds = layer(x_batch)
        # Same shapes as in the original implementation.
        self.assertEqual((batch_size, 2, hidden_layer_size), x_embeds.static.shape)
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size, 4),
            x_embeds.known.shape,
        )
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size, 1),
            x_embeds.observed.shape,
        )

    def test_static_covariates_encoder(self):
        layer = StaticCovariatesEncoder(
            hidden_layer_size, dropout_rate=0.0, prng_seed=PRNG_SEED
        )
        static_context = layer(tf.random.uniform((batch_size, 2, hidden_layer_size)))
        # Same shapes as in the original implementation.
        self.assertEqual((batch_size, hidden_layer_size), static_context.vector.shape)
        self.assertEqual(
            (batch_size, hidden_layer_size), static_context.enrichment.shape
        )
        self.assertEqual((batch_size, hidden_layer_size), static_context.state_h.shape)
        self.assertEqual((batch_size, hidden_layer_size), static_context.state_c.shape)

    @parameterized.parameters((25, 4), (5, 3))
    def test_temporal_variable_selection_network(self, time_steps, features):
        layer = TemporalVariableSelectionNetwork(
            hidden_layer_size, prng_seed=PRNG_SEED, dropout_rate=0
        )
        x = ContextAwareInputs(
            inputs=tf.random.uniform(
                (batch_size, time_steps, hidden_layer_size, features)
            ),
            context=tf.random.uniform((batch_size, hidden_layer_size)),
        )
        feature, flags, _ = layer(x)
        # Same shapes as in the original implementation.
        self.assertEqual((batch_size, time_steps, hidden_layer_size), feature.shape)
        self.assertEqual((batch_size, time_steps, 1, features), flags.shape)

    def test_decoder(self):
        layer = TemporalFusionDecoder(
            4, hidden_layer_size=hidden_layer_size, dropout_rate=0, prng_seed=PRNG_SEED
        )
        decoder_in = DecoderInputs(
            lstm_outputs=tf.random.uniform(
                (batch_size, n_time_steps, hidden_layer_size)
            ),
            input_embeddings=tf.random.uniform(
                (batch_size, n_time_steps, hidden_layer_size)
            ),
            context_vector=tf.random.uniform((batch_size, hidden_layer_size)),
        )
        decoder_out = layer(decoder_in)
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size),
            decoder_out.shape,
        )


class TFTModelTest(tf.test.TestCase, parameterized.TestCase):
    def test_tft_model(self):
        x_batch = make_x_batch()
        model = TemporalFusionTransformer(
            num_encoder_steps=25,
            num_attention_heads=4,
            hidden_layer_size=hidden_layer_size,
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=known_categories_sizes,
            dropout_rate=0,
            quantiles=[1, 2, 3],
            output_size=1,
        )
        logits = model(x_batch)
        tf.debugging.check_numerics(logits, "Test Failed.")
        # 3 is default number of quantiles.
        self.assertEqual((batch_size, 5, 3), logits.shape)

    @parameterized.named_parameters(
        (
            "known_categorical==None",
            TFTInputs(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
                known_categorical=None,
                observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
            ),
            [],
        ),
        (
            "observed==None",
            TFTInputs(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
                known_categorical=(tf.ones((8, 30, 2), dtype=tf.int32)),
                observed=None,
            ),
            known_categories_sizes,
        ),
        (
            "known_categorical==None, observed==None",
            TFTInputs(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
                known_categorical=None,
                observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
            ),
            [],
        ),
    )
    def test_input_is_missing(self, x_batch, _known_categories_sizes):
        model = TemporalFusionTransformer(
            num_encoder_steps=25,
            num_attention_heads=4,
            hidden_layer_size=hidden_layer_size,
            static_categories_sizes=static_categories_sizes,
            known_categories_sizes=_known_categories_sizes,
            dropout_rate=0,
            output_size=1,
        )

        logits = model(x_batch)
        tf.debugging.check_numerics(logits, "Test Failed.")
        # 3 is default number of quantiles.
        self.assertEqual((batch_size, 5, 3), logits.shape)


def make_x_batch():
    return TFTInputs(
        static=tf.ones((8, 4), dtype=tf.int32),
        known_real=tf.ones((8, 30, 3), dtype=tf.float32),
        known_categorical=tf.ones((8, 30, 2), dtype=tf.int32),
        observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
    )
