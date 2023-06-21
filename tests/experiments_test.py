import tensorflow as tf
import tempfile
from temporal_fusion_transformer.src.experiments import electricity_experiment


def load_batch(path: str):
    return tf.data.Dataset.load(path).batch(8).take(1).as_numpy_iterator().next()


class ExperimentsTest(tf.test.TestCase):
    def test_electricity(self):  # noqa
        with tempfile.TemporaryDirectory() as tmp_dir:
            electricity_experiment.process_raw_data(
                "tests/test_data/electricity/mini.csv", tmp_dir
            )

            train_data = load_batch(f"{tmp_dir}/electricity/training")
            validation_data = load_batch(f"{tmp_dir}/electricity/validation")
            test_data = load_batch(f"{tmp_dir}/electricity/test")

        assert "identifier" not in train_data
        assert "identifier" not in validation_data
        assert "identifier" in test_data

        assert "time" not in train_data
        assert "time" not in validation_data
        assert "time" in test_data
