import tensorflow as tf


class QuantileLossTest(tf.test.TestCase):
    quantiles = [0.1, 0.5, 0.9]

    def test_loss_obj(self):
        pass
