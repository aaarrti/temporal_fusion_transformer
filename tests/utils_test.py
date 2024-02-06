import numpy as np
from temporal_fusion_transformer.src.datasets.utils import time_series_to_array
from keras.utils import timeseries_dataset_from_array
import tensorflow as tf


def test_time_series_to_array():
    arr = np.arange(10, dtype=int).reshape((10, 1))
    ts: tf.data.Dataset = timeseries_dataset_from_array(arr, None, sequence_length=4)

    result = np.asarray(*ts.as_numpy_iterator())
    result = time_series_to_array(result)

    np.testing.assert_array_equal(arr, result)
