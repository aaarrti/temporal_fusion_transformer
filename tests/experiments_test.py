import tensorflow as tf
from temporal_fusion_transformer.src.datasets.electricity import Electricity


def test_electricity():
    electricity_experiment = Electricity()
    (ds, _, _), _ = electricity_experiment.make_dataset("tests/test_data/electricity/mini.csv")
    x, y = ds.batch(8).as_numpy_iterator().next()
    tf.debugging.assert_rank(x, 3)
    assert x.shape[1] == electricity_experiment.fixed_parameters.total_time_steps
