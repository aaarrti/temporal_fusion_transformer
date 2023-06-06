import numpy as np
import tensorflow as tf

from temporal_fusion_transformer.experiments import ElectricityExperiment
from temporal_fusion_transformer.experiments import map_dict


class ExperimentsTest(tf.test.TestCase):
    def test_electricity(self):  # noqa
        train_ds, val_ds, test_ds = ElectricityExperiment.from_raw_csv(
            "tests/assets/electricity/LD2011_2014_mini.txt"
        )

        def assert_electricity_data(data):
            assert isinstance(data, dict)
            expected_shapes = {
                "identifier": ((192, 1), object),
                "inputs_known_real": ((192, 3), float),
                "inputs_static": ((1,), int),
                "outputs": ((24, 1), float),
                "time": ((192, 1), float),
            }
            for k, (expected_shape, expected_dtype) in expected_shapes.items():
                arr = data[k]
                assert isinstance(arr, np.ndarray)
                inner_shape = arr.shape[1:]
                assert inner_shape == expected_shape
                assert arr.dtype == expected_dtype

        assert_electricity_data(train_ds)
        assert_electricity_data(val_ds)
        assert_electricity_data(test_ds)
        train_ds = map_dict(train_ds, lambda v: v[:8])
        val_ds = map_dict(val_ds, lambda v: v[:8])
        np.savez("tests/assets/electricity/train.npz", **train_ds)
        np.savez("tests/assets/electricity/validation.npz", **val_ds)
