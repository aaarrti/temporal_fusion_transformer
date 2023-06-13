import tensorflow as tf
from temporal_fusion_transformer.quantile_loss import QuantileLoss

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
