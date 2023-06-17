import numpy as np
import tensorflow as tf

from temporal_fusion_transformer.src.experiments import electricity_experiment


class ExperimentsTest(tf.test.TestCase):
    def test_electricity(self):  # noqa
        data, _ = electricity_experiment.read_raw_csv(
            "tests/assets/electricity/LD2011_2014_mini.csv"
        )

        def assert_electricity_data(data):
            assert isinstance(data, dict)
            expected_shapes = {
                "identifier": ((192, 1), object),
                "inputs_known_real": ((192, 3), np.float32),
                "inputs_static": ((1,), np.int32),
                "outputs": ((24, 1), np.float32),
                "time": ((192, 1), np.float32),
            }
            for k, (expected_shape, expected_dtype) in expected_shapes.items():
                arr = data[k]
                assert isinstance(arr, np.ndarray)
                inner_shape = arr.shape[1:]
                assert inner_shape == expected_shape
                assert arr.dtype == expected_dtype

        assert_electricity_data(data.train)
        assert_electricity_data(data.validation)
        assert_electricity_data(data.test)
        # train_ds = map_dict(train_ds, value_mapper=lambda v: v[:8])
        # val_ds = map_dict(val_ds, value_mapper=lambda v: v[:8])
        # np.savez("tests/assets/electricity/train.npz", **train_ds)
        # np.savez("tests/assets/electricity/validation.npz", **val_ds)
