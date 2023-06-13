import tensorflow as tf
from temporal_fusion_transformer.tf.quantile_loss import QuantileLoss
from tests.constants import QUANTILES


class QuantileLossTest(tf.test.TestCase):
    def setUp(self):
        self.loss_fn = QuantileLoss(QUANTILES)

    def test_loss_fn(self):
        y_true = tf.random.uniform((8, 24, 1))
        y_pred = tf.random.uniform((8, 24, 3))
        loss = self.loss_fn(y_true, y_pred)
        tf.debugging.check_numerics(loss, "Test Failed.")
        tf.debugging.assert_rank(loss, 0)

    def test_loss_converges_to_0(self):
        y_true = tf.ones((8, 24, 1))
        y_pred = tf.ones((8, 24, 3))

        loss = self.loss_fn(y_true, y_pred)
        tf.debugging.assert_rank(loss, 0)
        tf.debugging.assert_near(loss, 0.0)
