import tensorflow as tf
from temporal_fusion_transformer.tf.quantile_loss import (
    quantile_loss,
    quantile_rmse,
    QuantileLoss,
    QuantileRMSE,
)
from tests.constants import PRNG_SEED

quantiles = [0.1, 0.5, 0.9]


class QuantileLossTest(tf.test.TestCase):
    def test_loss_fn(self):
        y_true = tf.random.uniform((8, 24, 1), seed=PRNG_SEED)
        y_pred = tf.random.uniform((8, 24, 3), seed=PRNG_SEED)
        loss = quantile_loss(y_true, y_pred, tf.constant(quantiles))
        tf.debugging.check_numerics(loss, "Test Failed.")
        tf.debugging.assert_rank(loss, 1)
        assert loss.shape == (8,)

    def test_loss_converges_to_0(self):
        y_true = tf.ones((8, 24, 1))
        y_pred = tf.ones((8, 24, 3))

        loss = quantile_loss(y_true, y_pred, tf.constant(quantiles))
        tf.debugging.assert_rank(loss, 1)
        assert loss.shape == (8,)
        tf.debugging.assert_near(loss, 0.0)

    def test_loss_fn_wrapper(self):
        y_true = tf.random.uniform((8, 24, 1), seed=PRNG_SEED)
        y_pred = tf.random.uniform((8, 24, 3), seed=PRNG_SEED)

        loss = QuantileLoss()(y_true, y_pred)
        tf.debugging.assert_rank(loss, 0)
        tf.debugging.check_numerics(loss, "Test Failed.")


class QuantileRMSETest(tf.test.TestCase):
    def test_rmse(self):
        y_true = tf.random.uniform((8, 24, 1), seed=PRNG_SEED)
        y_pred = tf.random.uniform((8, 24, 3), seed=PRNG_SEED)
        rmse = quantile_rmse(y_true, y_pred, tf.constant(quantiles))
        tf.debugging.check_numerics(rmse, "Test Failed.")
        tf.debugging.assert_rank(rmse, 1)
        assert rmse.shape == (8,)

    def test_rmse_converges_to_0(self):
        y_true = tf.ones((8, 24, 1))
        y_pred = tf.ones((8, 24, 3))

        rmse = quantile_rmse(y_true, y_pred, tf.constant(quantiles))
        tf.debugging.assert_rank(rmse, 1)
        assert rmse.shape == (8,)
        tf.debugging.assert_near(rmse, 0.0)

    def test_metric_wrapper(self):
        y_true = tf.random.uniform((8, 24, 1), seed=PRNG_SEED)
        y_pred = tf.random.uniform((8, 24, 3), seed=PRNG_SEED)
        rmse = QuantileRMSE()(y_true, y_pred)
        tf.debugging.check_numerics(rmse, "Test Failed.")
        tf.debugging.assert_rank(rmse, 0)
