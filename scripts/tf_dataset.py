from __future__ import annotations

import tensorflow as tf

from keras.utils import FeatureSpace
from absl import flags, logging
from absl_extra import tasks, logging_utils
import temporal_fusion_transformer as tft

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", help="Data directory", default="data")
logging_utils.setup_logging(log_level="DEBUG")


@tasks.register_task(name="electricity")
def make_electricity_dataset_task():
    data_dir = f"{FLAGS.data_dir}/electricity"
    (training_dataset, validation_dataset, test_dataset), feature_space = tft.datasets.Electricity().make_dataset(data_dir)
    persist_dataset(training_dataset, validation_dataset, test_dataset, feature_space, data_dir)


@tasks.register_task(name="favorita")
def make_favorita_dataset_task():
    data_dir = f"{FLAGS.data_dir}/electricity"
    (training_dataset, validation_dataset, test_dataset), feature_space = tft.datasets.Favorita().make_dataset(data_dir)
    persist_dataset(training_dataset, validation_dataset, test_dataset, feature_space, data_dir)


def persist_dataset(
        training_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        feature_space: FeatureSpace,
        data_dir: str
):
    logging.info(f"Saving training split")
    training_dataset.save(f"{data_dir}/training", compression="GZIP")
    logging.info(f"Saving validation split")
    validation_dataset.save(f"{data_dir}/validation", compression="GZIP")
    logging.info(f"Saving test split")
    test_dataset.save(f"{data_dir}/test", compression="GZIP")
    feature_space.save(f"{data_dir}/features_space.keras")


if __name__ == "__main__":
    """Run with e.g., `python scripts/tf_dataset.py --experiment=electricity`"""
    tasks.run(task_flag="experiment")
    
    
