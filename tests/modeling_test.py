import tensorflow as tf
from absl.testing import parameterized
from keras.utils.tf_utils import can_jit_compile

from temporal_fusion_transformer.src.experiments import electricity_experiment
from temporal_fusion_transformer.src.modeling import (
    TemporalFusionTransformer,
    StaticCovariatesEncoder,
    TFTInputEmbedding,
    VariableSelection,
    EncoderBlock,
    ContextInputs,
)
from temporal_fusion_transformer.src.utils import make_tft_model
from tests.constants import PRNG_SEED

static_categories_sizes = [2, 2]
known_categories_sizes = [4]
n_time_steps = 30
batch_size = 8
hidden_layer_size = 5


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
        self.assertEqual((batch_size, 2, hidden_layer_size), x_embeds["static"].shape)
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size, 4),
            x_embeds["known"].shape,
        )
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size, 1),
            x_embeds["observed"].shape,
        )

    def test_static_covariates_encoder(self):
        layer = StaticCovariatesEncoder(
            hidden_layer_size, dropout_rate=0.0, prng_seed=PRNG_SEED
        )
        static_context = layer(tf.random.uniform((batch_size, 2, hidden_layer_size)))
        # Same shapes as in the original implementation.
        self.assertEqual(
            (batch_size, hidden_layer_size), static_context["vector"].shape
        )
        self.assertEqual(
            (batch_size, hidden_layer_size), static_context["enrichment"].shape
        )
        self.assertEqual(
            (batch_size, hidden_layer_size), static_context["state_h"].shape
        )
        self.assertEqual(
            (batch_size, hidden_layer_size), static_context["state_c"].shape
        )

    @parameterized.parameters((25, 4), (5, 3))
    def test_variable_selection_network(self, time_steps, features):
        layer = VariableSelection(
            hidden_layer_size, prng_seed=PRNG_SEED, dropout_rate=0
        )
        x = ContextInputs(
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
        layer = EncoderBlock(
            4, hidden_layer_size=hidden_layer_size, dropout_rate=0, prng_seed=PRNG_SEED
        )
        decoder_out = layer(
            tf.random.uniform(
                (batch_size, n_time_steps, hidden_layer_size), seed=PRNG_SEED
            ),
        )
        self.assertEqual(
            (batch_size, n_time_steps, hidden_layer_size),
            decoder_out.shape,
        )


class TFTModelTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(1, 4, 12)
    def test_tft_model(self, num_stacks):
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
            num_stacks=num_stacks,
        )
        logits = model(x_batch)
        tf.debugging.check_numerics(logits, "Test Failed.")
        # 3 is default number of quantiles.
        self.assertEqual((batch_size, 5, 3), logits.shape)

    @parameterized.named_parameters(
        (
            "known_categorical==None",
            dict(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
                observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
            ),
            [],
        ),
        (
            "observed==None",
            dict(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
                known_categorical=(tf.ones((8, 30, 2), dtype=tf.int32)),
            ),
            known_categories_sizes,
        ),
        (
            "known_categorical==None, observed==None",
            dict(
                static=tf.ones((8, 4), dtype=tf.int32),
                known_real=tf.ones((8, 30, 3), dtype=tf.float32),
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
    return dict(
        static=tf.ones((8, 4), dtype=tf.int32),
        known_real=tf.ones((8, 30, 3), dtype=tf.float32),
        known_categorical=tf.ones((8, 30, 2), dtype=tf.int32),
        observed=tf.random.uniform((8, 30, 1), dtype=tf.float32),
    )


class TrainStepTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.train_ds = (
            tf.data.Dataset.load("tests/test_data/electricity/mini_dataset")
            .batch(8)
            .take(1)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val_ds = (
            tf.data.Dataset.load("tests/test_data/electricity/mini_dataset")
            .batch(8)
            .skip(1)
            .take(1)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    @parameterized.parameters(
        [
            "float32",
            "float16",
            "mixed_float16",
            "mixed_bfloat16",
            "float16",
            "bfloat16",
        ]
    )
    def test_electricity(self, policy):
        tf.keras.mixed_precision.set_global_policy(policy)
        train_ds = self.train_ds.map(
            lambda i: make_input_tuple(
                i, tf.keras.mixed_precision.global_policy().compute_dtype, 24
            )
        )
        val_ds = self.val_ds.map(
            lambda i: make_input_tuple(
                i, tf.keras.mixed_precision.global_policy().compute_dtype, 24
            )
        )
        model = make_tft_model(
            electricity_experiment,
        )
        model.compile(tf.keras.optimizers.Adam(jit_compile=can_jit_compile(True)))
        history = model.fit(train_ds, validation_data=val_ds).history

        assert "val_loss" in history
        tf.debugging.check_numerics(history["val_loss"], "Test Failed.")

        assert "loss" in history
        tf.debugging.check_numerics(history["loss"], "Test Failed.")


def make_input_tuple(data, dtype, n):
    return (
        dict(
            static=data["inputs_static"],
            known_real=tf.cast(data["inputs_known_real"], dtype),
        ),
        tf.cast(data["outputs"][:, :n], dtype),
    )
