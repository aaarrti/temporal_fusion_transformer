import tensorflow as tf
from absl.testing import parameterized
from keras.utils.tf_utils import can_jit_compile

from temporal_fusion_transformer.experiments import (
    ElectricityExperiment,
    ModelParams,
    DataParams,
)
from temporal_fusion_transformer.modeling import TemporalFusionTransformer
from temporal_fusion_transformer.train_lib import QuantileLoss, load_data_from_archive

tf.config.run_functions_eagerly(True)


class QuantileLossTest(tf.test.TestCase):
    def setUp(self):
        self.quantiles = [0.1, 0.5, 0.9]
        self.loss_fn = QuantileLoss(self.quantiles)

    def test_loss_fn(self):
        y_true = tf.random.uniform((8, 24, 1))
        y_pred = tf.random.uniform((8, 24, 3))
        loss = self.loss_fn(y_true, y_pred)
        tf.debugging.check_numerics(loss, "Test Failed.")
        tf.debugging.assert_rank(loss, 0)


class TrainStepTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        train_ds = load_data_from_archive("tests/assets/electricity/train.npz")
        val_ds = load_data_from_archive("tests/assets/electricity/validation.npz")

        self.train_ds = tf.data.Dataset.from_tensors(train_ds)
        self.val_ds = tf.data.Dataset.from_tensors(val_ds)

    @parameterized.parameters(
        [
            (
                "float32",
                False,
            ),
            (
                "float16",
                False,
            ),
            (
                "mixed_float16",
                False,
            ),
            (
                "mixed_bfloat16",
                False,
            ),
            (
                "float16",
                False,
            ),
            (
                "bfloat16",
                False,
            ),
            (
                "float32",
                True,
            ),
            (
                "float16",
                True,
            ),
            (
                "mixed_float16",
                True,
            ),
            (
                "mixed_bfloat16",
                True,
            ),
            (
                "float16",
                True,
            ),
            (
                "bfloat16",
                True,
            ),
        ]
    )
    def test_train_step(self, policy, unroll_lstm):
        tf.keras.mixed_precision.set_global_policy(policy)
        train_ds = self.train_ds.map(
            lambda i: make_input_tuple(
                i, tf.keras.mixed_precision.global_policy().compute_dtype
            )
        )
        val_ds = self.val_ds.map(
            lambda i: make_input_tuple(
                i, tf.keras.mixed_precision.global_policy().compute_dtype
            )
        )

        hp: ModelParams = ElectricityExperiment.default_params[0]
        fp: DataParams = ElectricityExperiment.fixed_params

        model = TemporalFusionTransformer(
            static_categories_sizes=fp.static_categories_sizes,
            known_categories_sizes=fp.known_categories_sizes,
            num_encoder_steps=fp.num_encoder_steps,
            hidden_layer_size=hp.hidden_layer_size,
            num_attention_heads=hp.num_attention_heads,
            unroll_lstm=unroll_lstm,
        )
        model.compile(
            tf.keras.optimizers.Adam(jit_compile=can_jit_compile()),
            loss=QuantileLoss(model.quantiles),
            metrics=[],
            jit_compile=can_jit_compile(),
        )
        history = model.fit(train_ds, validation_data=val_ds).history

        assert "val_loss" in history
        tf.debugging.check_numerics(history["val_loss"], "Test Failed.")

        assert "loss" in history
        tf.debugging.check_numerics(history["loss"], "Test Failed.")


def make_input_tuple(data, dtype):
    return (
        dict(
            static=data["inputs_static"],
            known_real=tf.cast(data["inputs_known_real"], dtype),
        ),
        tf.cast(data["outputs"], dtype),
    )
