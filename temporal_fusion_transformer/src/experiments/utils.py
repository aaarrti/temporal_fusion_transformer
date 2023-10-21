from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal

import numpy as np
from keras_core.utils import timeseries_dataset_from_array
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

    from temporal_fusion_transformer.src.experiments.base import Preprocessor


def time_series_dataset_from_dataframe(
    df: pl.DataFrame,
    inputs: List[str],
    targets: List[str],
    total_time_steps: int,
    id_column: str,
    preprocessor: Preprocessor | None = None,
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
    preprocessor

    Returns
    -------

    tf_ds:
        Not batched tf.data.Dataset

    """

    if preprocessor is not None:
        df = preprocessor(df)

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
        time_series = time_series.map(
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
            df.groupby(id_column), total=num_groups, desc="Converting to time-series dataset"
        ):
            try:
                time_series_i = make_time_series_fn(df_i)
                num_ok += 1
                yield time_series_i
            except ValueError as e:
                log.error(e)
                num_errors += 1
        log.info(f"{num_errors = }, {num_ok = }")

    return functoolz.reduce(lambda a, b: a.concatenate(b), generator())


def time_series_to_array(ts: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    ts:
        2D time series, or 3D batched time series.

    Returns
    -------

    arr:
        2D array, without repeated instances.

    """
    if np.ndim(ts) != 3:
        raise ValueError("ts must be a 2D or 3d array")

    first_ts = ts[0, :-1]
    rest = [i[-1] for i in ts]
    return np.concatenate([first_ts, rest], axis=0)


def persist_dataset(
    training_ds: tf.data.Dataset,
    validation_ds: tf.data.Dataset,
    test_df: pl.DataFrame,
    preprocessor: Preprocessor,
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
    preprocessor
    save_dir
    compression
    test_split_save_format:
        By default will save test split in Parquet format. For smaller dataset you can use CSV though.

    Returns
    -------

    """

    log.info("Saving (preprocessed) train split")
    training_ds.save(f"{save_dir}/training", compression=compression)
    log.info("Saving (preprocessed) validation split")
    validation_ds.save(f"{save_dir}/validation", compression=compression)
    log.info("Saving (not preprocessed) test split (as parquet)")
    if test_split_save_format == "parquet":
        test_df.write_parquet(f"{save_dir}/test.parquet")
    if test_split_save_format == "csv":
        test_df.write_csv(f"{save_dir}/test.csv")

    log.info("Saving preprocessor state")
    preprocessor.save(f"{save_dir}/preprocessor.keras")


def count_groups(df: pl.DataFrame, id_column: str) -> int:
    return len(df[id_column].unique_counts())
