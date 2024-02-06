from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from keras.utils import timeseries_dataset_from_array
from toolz import functoolz

log = logging.getLogger(__name__)

try:
    import polars as pl
    import tensorflow as tf
    from tqdm.auto import tqdm
except ModuleNotFoundError as ex:
    log.warning(ex)


if TYPE_CHECKING:
    import polars as pl


def time_series_dataset_from_dataframe(
    df: pl.DataFrame,
    inputs: list[str],
    targets: list[str],
    total_time_steps: int,
    id_column: str,
) -> tf.data.Dataset:
    """

    - Group by `id_column`
    - apply `preprocess_fn`
    - apply `keras.utils.timeseries_dataset_from_array`
    - join groups in 1 TF dataset

    Parameters
    ----------
    df
    inputs
    targets
    total_time_steps
    id_column

    Returns
    -------

    tf_ds:
        Not batched tf.data.Dataset

    """

    report_columns_mismatch(df, targets + inputs + [id_column])

    # for some reason, keras would generate targets of shape [1, n] and inputs [time_steps, n],
    # but we need time-steps for y_batch also, we need is [time_steps, m]. We don't need `sequence_stride`,
    # since we don't want any synthetic repetitions.
    # -1 for TARGETS
    num_inputs = len(inputs)
    num_groups = count_groups(df, id_column)

    def make_time_series_fn(sub_df: pl.DataFrame) -> tf.data.Dataset:
        arr: np.ndarray = sub_df[inputs + targets].to_numpy(order="c")

        if len(arr) < total_time_steps:
            raise ValueError("len(arr) < total_time_steps")

        time_series: tf.data.Dataset = timeseries_dataset_from_array(
            arr,
            targets=None,
            sequence_length=total_time_steps,
            batch_size=None,
        )
        time_series = time_series.map(lambda i: tf.cast(i, tf.float32)).map(
            lambda x: (x[..., :num_inputs], x[..., num_inputs:]),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        return time_series

    if num_groups == 1:
        return make_time_series_fn(df)

    def generator():
        num_errors = 0
        num_ok = 0
        for _, df_i in tqdm(
            df.groupby(id_column, maintain_order=True),
            total=num_groups,
            desc="Converting to time-series dataset",
        ):
            try:
                time_series_i = make_time_series_fn(df_i)
                num_ok += 1
                yield time_series_i
            except ValueError as e:
                # log.error(e)
                num_errors += 1
        log.info(f"{num_errors = }, {num_ok = }")

    def concat_datasets(a: tf.data.Dataset, b: tf.data.Dataset) -> tf.data.Dataset:
        return a.concatenate(b)

    return functoolz.reduce(concat_datasets, generator())


def dataframe_from_time_series_dataset(
    ds: tf.data.Dataset,
    inputs_mappings: dict[str, int],
    targets_mappings: dict[str, int],
    inverse_preprocess: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    x = np.asarray([i for i, _ in ds.as_numpy_iterator()])
    y = np.asarray([j for _, j in ds.as_numpy_iterator()])

    x_flat = time_series_to_array(x)
    y_flat = time_series_to_array(y)

    inputs = {k: x_flat[..., v] for k, v in inputs_mappings.items()}
    targets = {k: y_flat[..., v] for k, v in targets_mappings.items()}
    df = pl.DataFrame({**inputs, **targets})
    if inverse_preprocess is not None:
        df = inverse_preprocess(df)

    return df


def time_series_to_array(ts: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    ts:
        3D time series.

    Returns
    -------

    arr:
        2D array, without repeated instances.

    """
    if np.ndim(ts) != 3:
        raise ValueError("ts must be 3d array")

    first_ts = ts[0, :-1]
    rest = [i[-1] for i in ts]
    return np.concatenate([first_ts, rest], axis=0)


def persist_dataset(
    training_ds: tf.data.Dataset,
    validation_ds: tf.data.Dataset,
    test_df: pl.DataFrame,
    save_dir: str,
    compression: Literal["GZIP"] | None = "GZIP",
    test_split_save_format: Literal["csv", "parquet"] = "parquet",
):
    """

    Parameters
    ----------
    training_ds
    validation_ds
    test_df
    save_dir
    compression
    test_split_save_format:
        By default will save test split in Parquet format. For smaller dataset you can use CSV though.

    Returns
    -------

    """

    if training_ds.cardinality() == 0:
        raise ValueError("training_ds.cardinality() == 0")

    if validation_ds.cardinality() == 0:
        raise ValueError("validation_ds.cardinality() == 0")

    if len(test_df) == 0:
        raise ValueError("len(test_df) == 0")

    log.info("Saving (preprocessed) train split")
    training_ds.save(f"{save_dir}/training", compression=compression)
    log.info("Saving (preprocessed) validation split")
    validation_ds.save(f"{save_dir}/validation", compression=compression)
    log.info(f"Saving (not preprocessed) test split (as {test_split_save_format})")
    if test_split_save_format == "parquet":
        test_df.write_parquet(f"{save_dir}/test.parquet")
    if test_split_save_format == "csv":
        test_df.write_csv(f"{save_dir}/test.csv")


def count_groups(df: pl.DataFrame, id_column: str) -> int:
    return len(df[id_column].unique_counts())


def report_columns_mismatch(df: pl.DataFrame, expected_columns: list[str]):
    expected_columns = set(expected_columns)
    found_columns = set(df.columns)

    unexpected_cols = found_columns.difference(expected_columns)
    missing_cols = expected_columns.difference(found_columns)

    if len(unexpected_cols) > 0:
        log.error(f"Found unexpected columns: {tuple(unexpected_cols)}")

    if len(missing_cols) > 0:
        log.error(f"Missing expected columns: {tuple(missing_cols)}")
