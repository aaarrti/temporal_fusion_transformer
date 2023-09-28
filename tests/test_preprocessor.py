from pathlib import Path

import chex
import jax
import numpy as np
import unittest
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import polars as pl
import polars.testing
import tensorflow as tf

from absl_extra.flax_utils import load_from_msgpack

from temporal_fusion_transformer.src.experiments.util import (
    deserialize_preprocessor,
    serialize_preprocessor,
)
from keras.preprocessing import timeseries_dataset_from_array
from temporal_fusion_transformer.src.experiments.util import time_series_to_array
from temporal_fusion_transformer.src.experiments.electricity import (
    inverse_transform_for_single_id,
    DataPreprocessor,
    inverse_transform,
    apply_preprocessor,
    restore_timestamps,
)
from temporal_fusion_transformer.src.experiments.electricity import make_time_series_dataset


def test_serialize_preprocessor(tmp_path: Path):
    x = jax.random.normal(jax.random.PRNGKey(33), (120, 4))
    x = np.asarray(x)
    labels = [str(i) for i in range(12)]

    sc = StandardScaler()
    le = LabelEncoder()

    x_transformed = sc.fit_transform(x)
    label_transformed = le.fit_transform(labels)

    preprocessor = {"target": sc, "categorical": le}

    serialize_preprocessor(preprocessor, tmp_path.as_posix())

    reloaded_preprocessor = deserialize_preprocessor(tmp_path.as_posix())

    x_transformed_2 = reloaded_preprocessor["target"].transform(x)
    label_transformed_2 = reloaded_preprocessor["categorical"].transform(labels)

    np.testing.assert_allclose(x_transformed, x_transformed_2)
    np.testing.assert_equal(label_transformed, label_transformed_2)


@pytest.fixture(scope="module")
def inference_data():
    return load_from_msgpack(None, "tests/test_data/electricity_inference_data.msgpack")


@pytest.fixture(scope="module")
def mt_124_df():
    return pl.read_parquet("tests/test_data/mt_124_test.parquet")


@pytest.fixture(scope="module")
def preprocessor_dict():
    return deserialize_preprocessor("tests/test_data/electricity_preprocessor.msgpack")


def test_inverse_transform(inference_data, preprocessor_dict):
    x_batch = inference_data["x_batch"].reshape((-1, 6))
    y_batch = inference_data["y_batch"].reshape((-1, 1))
    df = inverse_transform_for_single_id(preprocessor_dict, x_batch, y_batch, inference_data["id"])
    print()


@pytest.mark.parametrize("shape, num_time_steps", [((100, 1), 10), ((337, 6), 192)])
def test_make_time_series(shape, num_time_steps):
    x = np.ones(shape)
    # Super memory greedy original implementation
    ts1 = np.stack([x[i : len(x) - (num_time_steps - 1) + i, :] for i in range(num_time_steps)], axis=1)

    tf_ds: tf.data.Dataset = timeseries_dataset_from_array(x, None, num_time_steps, batch_size=None)
    ts2 = np.asarray(list(tf_ds.as_numpy_iterator()))

    assert ts1.shape == ts2.shape
    np.testing.assert_equal(ts1, ts2)


@pytest.mark.parametrize("shape, num_time_steps", [((100, 1), 10), ((100, 1), 3), ((337, 6), 192)])
def test_time_series_to_array(shape, num_time_steps):
    x = np.random.uniform(size=shape)
    ts = np.asarray(list(timeseries_dataset_from_array(x, None, num_time_steps, batch_size=None).as_numpy_iterator()))

    x_restored = time_series_to_array(ts)

    assert x.shape == x_restored.shape
    np.testing.assert_equal(x, x_restored)


def append_time(df: pl.DataFrame):
    ts = restore_timestamps(df)
    return df.with_columns(time=pl.lit(ts)).sort(by="time")


def test_transform_and_inverse_transform(mt_124_df, preprocessor_dict):
    tf_ds = make_time_series_dataset(mt_124_df, preprocessor_dict).batch(8)

    x_batch = np.concatenate(list(tf_ds.map(lambda x, y: x).as_numpy_iterator()), axis=0)
    y_batch = np.concatenate(list(tf_ds.map(lambda x, y: y).as_numpy_iterator()), axis=0)

    restored_df = inverse_transform(
        preprocessor_dict,
        time_series_to_array(x_batch),
        time_series_to_array(y_batch),
    )

    unittest.TestCase().assertCountEqual(mt_124_df.columns, restored_df.columns)

    mt_124_df = append_time(mt_124_df)
    restored_df = append_time(restored_df)

    polars.testing.assert_frame_equal(mt_124_df, restored_df, check_column_order=False, atol=0.01)
