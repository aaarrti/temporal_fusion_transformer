import tensorflow as tf

from temporal_fusion_transformer.plotting import plot_predictions
from tests.constants import PRNG_SEED


class PlottingTest(tf.test.TestCase):
    def setUp(self):
        self.past_time = tf.random.uniform(shape=[8, 20], seed=PRNG_SEED)
        self.future_time = tf.random.uniform(shape=[8, 10], seed=PRNG_SEED)
        self.past_outputs = tf.random.uniform(shape=[8, 20, 2], seed=PRNG_SEED)
        self.future_outputs = tf.random.uniform(shape=[8, 10, 2], seed=PRNG_SEED)
        self.predicted_outputs = tf.random.uniform(shape=[8, 10, 6], seed=PRNG_SEED)

    def test_plot_full(self):
        plot_predictions(
            predicted_outputs=self.predicted_outputs,
            future_timestamps=self.future_time,
            num_outputs=2,
            quantiles=[0.1, 0.5, 0.9],
            past_outputs=self.past_outputs,
            past_time_stamps=self.past_time,
            future_outputs=self.future_outputs,
        )
