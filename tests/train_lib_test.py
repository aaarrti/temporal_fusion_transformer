from __future__ import annotations

import tensorflow as tf

from temporal_fusion_transformer.experiments import (
    ElectricityExperiment,
    ModelParams,
    DataParams,
)
from temporal_fusion_transformer.modeling import TemporalFusionTransformer
from temporal_fusion_transformer.train_lib import (
    QuantileLoss,
    train_with_fixed_hyper_parameters,
    load_data_from_archive,
    make_input_tuple,
)

# tf.config.run_functions_eagerly(True)


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


class TrainStepTest(tf.test.TestCase):
    def setUp(self):
        train_ds = load_data_from_archive("tests/assets/electricity/train.npz")
        val_ds = load_data_from_archive("tests/assets/electricity/validation.npz")

        self.train_ds = tf.data.Dataset.from_tensors(train_ds).map(make_input_tuple)
        self.val_ds = tf.data.Dataset.from_tensors(val_ds).map(make_input_tuple)

    def test_train_step(self):
        hp: ModelParams = ElectricityExperiment.default_params[0]
        fp: DataParams = ElectricityExperiment.fixed_params

        def make_model():
            return TemporalFusionTransformer(
                static_categories_sizes=fp.static_categories_sizes,
                known_categories_sizes=fp.known_categories_sizes,
                num_encoder_steps=fp.num_encoder_steps,
                hidden_layer_size=hp.hidden_layer_size,
                num_attention_heads=hp.num_attention_heads,
            )

        _, history = train_with_fixed_hyper_parameters(
            make_model,
            lambda: "adam",
            self.train_ds,
            self.val_ds,
        )

        assert "val_loss" in history
        tf.debugging.check_numerics(history["val_loss"], "Test Failed.")

        assert "loss" in history
        tf.debugging.check_numerics(history["loss"], "Test Failed.")
