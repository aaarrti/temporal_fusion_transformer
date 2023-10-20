from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import Tuple, Type

import keras_core as keras
import polars as pl
import tensorflow as tf
from keras.mixed_precision import global_policy
from keras_core import layers
from tqdm.auto import tqdm

from temporal_fusion_transformer.src import training
from temporal_fusion_transformer.src.experiments.base import (
    Experiment,
    MultiHorizonTimeSeriesDataset,
    Preprocessor,
)
from temporal_fusion_transformer.src.experiments.config import get_config
from temporal_fusion_transformer.src.experiments.utils import (
    count_groups,
    persist_dataset,
    time_series_dataset_from_dataframe,
)
from temporal_fusion_transformer.src.utils import classproperty

_ID_COLUMN = "id"
_REAL_INPUTS = ["year"]
_CATEGORICAL_INPUTS = ["month", "day", "hour", "day_of_week"]
_INPUTS = _REAL_INPUTS + _CATEGORICAL_INPUTS + [_ID_COLUMN]
_TARGETS = ["power_usage"]
log = logging.getLogger(__name__)


# ------ We export class based API for convenience ------------


class Electricity(Experiment):
    @classproperty
    def dataset(self) -> Type[ElectricityDataset]:
        return ElectricityDataset

    @classproperty
    def preprocessor(self) -> Type[ElectricityPreprocessor]:
        return ElectricityPreprocessor

    @staticmethod
    def train_model(
        data_dir: str = "data/electricity",
        batch_size: int = 128,
        epochs: int = 1,
        save_filename: str | None = "data/electricity/model.keras",
        **kwargs,
    ) -> keras.Model | None:
        config = get_config("electricity")
        dataset = training.load_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            num_encoder_steps=config.data.num_encoder_steps,
            dtype=global_policy().compute_dtype,
        )
        return training.train_model(
            dataset=dataset, epochs=epochs, save_filename=save_filename, config=config, **kwargs
        )

    @staticmethod
    def train_model_distributed(
        data_dir: str = "data", batch_size=128, epochs: int = 1, save_filename: str | None = "model.keras", **kwargs
    ):
        pass


class ElectricityDataset(MultiHorizonTimeSeriesDataset):
    def __init__(
        self,
        validation_boundary: datetime = datetime(2014, 8, 8),
        test_boundary: int = datetime(2014, 9, 1),
        split_overlap_days: int = 7,
        cutoff_days: Tuple[datetime, datetime] = (datetime(2014, 1, 1), datetime(2014, 9, 8)),
    ):
        """
        References
        ----------

        - https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

        Parameters
        ----------
        validation_boundary:
            Timestamp, after which data is considered validation split.
        test_boundary:
            Timestamp, after which data is considered test split.
        split_overlap_days:
            Number of days, which are overlapped between splits.
        cutoff_days:
            Tuple of start and end dates, before/after which data is not included.
        """
        self.validation_boundary = validation_boundary
        self.test_boundary = test_boundary
        self.cutoff_days = cutoff_days
        config = get_config("electricity").data
        self.total_time_steps = config.total_time_steps
        self.num_encoder_steps = config.num_encoder_steps
        self.split_overlap_days = split_overlap_days

    def convert_to_parquet(self, download_dir: str, output_dir: str | None = None, delete_processed: bool = True):
        return convert_to_parquet(download_dir, output_dir, delete_processed=delete_processed)

    def make_dataset(
        self, data_dir: str, save_dir: str | None = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, pl.DataFrame, Preprocessor]:
        df = read_parquet(data_dir, cutoff_days=self.cutoff_days)
        log.info(f"{df.columns = }")

        preprocessor = ElectricityPreprocessor.from_dataframe(df)
        preprocessor.adapt(df)

        training_df, validation_df, test_df = split_data(
            df, self.validation_boundary, self.test_boundary, self.split_overlap_days
        )

        kwargs = {
            "inputs": _INPUTS,
            "targets": _TARGETS,
            "total_time_steps": self.total_time_steps,
            "id_column": "id",
            "preprocessor": preprocessor,
        }
        training_time_series = time_series_dataset_from_dataframe(training_df, **kwargs)
        validation_time_series = time_series_dataset_from_dataframe(validation_df, **kwargs)

        if save_dir is not None:
            persist_dataset(training_time_series, validation_time_series, test_df, preprocessor, save_dir)
        else:
            return training_time_series, validation_time_series, test_df, preprocessor

    # def plot_predictions(
    #    self,
    #    df: pl.DataFrame,
    #    entity: str,
    #    preprocessor: Preprocessor,
    #    model: PredictFn,
    #    batch_size: int = 32,
    #    truncate_past: datetime | None = datetime(2014, 9, 4),
    # ) -> plt.Figure:
    #    from temporal_fusion_transformer.src.inference import plotting
    #    from temporal_fusion_transformer.src.modeling.tft_model import TftOutputs
    #
    #    df = df.filter(pl.col("id") == entity).pipe(
    #        lambda i: i.with_columns(timestamp=pl.Series(preprocessor.restore_timestamps(i)))
    #    )
    #    max_date = df["timestamp"].max()
    #
    #    df = df.filter(pl.col("timestamp") >= (max_date - timedelta(hours=self.total_time_steps))).drop("timestamp")
    #
    #    tf_ds = preprocessor.convert_dataframe_to_tf_dataset(df)
    #
    #    x = []
    #    y = []
    #    y_predicted = []
    #
    #    num_batches = ceil(float(tf_ds.cardinality()) / batch_size)
    #
    #    for x_batch, y_batch in tqdm(
    #        tf_ds.batch(batch_size).as_numpy_iterator(), total=num_batches, desc="Predicting..."
    #    ):
    #        y_predicted_batch = model(x_batch)
    #        if isinstance(y_predicted_batch, TftOutputs):
    #            y_predicted_batch = y_predicted_batch.logits
    #        y_predicted.extend(y_predicted_batch)
    #        x.extend(x_batch)
    #        y.extend(y_batch)
    #
    #    x = np.asarray(x)
    #    y = np.asarray(y)
    #    y_predicted = np.asarray(y_predicted)
    #
    #    ground_truth_df = preprocessor.inverse_transform(time_series_to_array(x), time_series_to_array(y))
    #    ts = preprocessor.restore_timestamps(ground_truth_df)
    #    pu = ground_truth_df["power_usage"].to_numpy()
    #
    #    quantiles = fixed_parameters.get_config("electricity").quantiles
    #    num_encoder_steps = fixed_parameters.get_config("electricity").num_encoder_steps
    #
    #    predicted_df = [
    #        preprocessor.inverse_transform(
    #            time_series_to_array(x[:, num_encoder_steps:]), time_series_to_array(y_predicted[..., i])
    #        )
    #        for i in range(len(quantiles))
    #    ]
    #
    #    ground_truth_past_ts = ts[:num_encoder_steps]
    #    ground_truth_past_pu = pu[:num_encoder_steps]
    #
    #    ground_truth_past_ts_truncated = []
    #    ground_truth_past_pu_truncated = []
    #
    #    for i, j in zip(ground_truth_past_ts, ground_truth_past_pu):
    #        if truncate_past is None or i >= truncate_past:
    #            ground_truth_past_ts_truncated.append(i)
    #            ground_truth_past_pu_truncated.append(j)
    #
    #    return plotting.plot_predictions(
    #        ground_truth_past=(ground_truth_past_ts_truncated, ground_truth_past_pu_truncated),
    #        ground_truth_observed=(ts[num_encoder_steps:], pu[num_encoder_steps:]),
    #        predictions=[(preprocessor.restore_timestamps(i), i["power_usage"].to_numpy()) for i in predicted_df],
    #        title=f"{entity} power usage".title(),
    #        quantiles=quantiles,
    #        formatter=mdates.DateFormatter("%m.%d %H"),
    #        locator=mdates.HourLocator(interval=20),
    #    )

    # def plot_feature_importance(
    #    self,
    #    df: pl.DataFrame,
    #    entity: str,
    #    preprocessor: Preprocessor,
    #    model: PredictFn,
    #    batch_size: int = 32,
    #    truncate_past: datetime | None = datetime(2014, 9, 4),
    # ) -> plt.Figure:
    #    fig = plt.figure(figsize=(7, 4))
    #
    #    df = df.filter(pl.col("id") == entity).pipe(
    #        lambda i: i.with_columns(timestamp=pl.Series(preprocessor.restore_timestamps(i)))
    #    )
    #    max_date = df["timestamp"].max()
    #
    #    df = df.filter(pl.col("timestamp") >= (max_date - timedelta(hours=self.total_time_steps))).drop("timestamp")
    #
    #    tf_ds = preprocessor.convert_dataframe_to_tf_dataset(df)
    #
    #    x = []
    #    y = []
    #
    #    num_batches = ceil(float(tf_ds.cardinality()) / batch_size)
    #
    #    for x_batch, y_batch in tqdm(
    #        tf_ds.batch(batch_size).as_numpy_iterator(), total=num_batches, desc="Predicting..."
    #    ):
    #        y_predicted_batch = model(x_batch)
    #        x.extend(x_batch)
    #        y.append(y_predicted_batch)
    #
    #    x = np.asarray(x)
    #    y = tree_map(lambda *args: jnp.concatenate([*args], axis=0), *y)
    #
    #    for i in range(len(_REAL_INPUTS + _CATEGORICAL_INPUTS)):
    #        past_df = preprocessor.inverse_transform(
    #            time_series_to_array(x[:, : self.num_encoder_steps]),
    #            time_series_to_array(y.historical_flags[..., i][..., None]),
    #        )
    #        future_df = preprocessor.inverse_transform(
    #            time_series_to_array(x[:, self.num_encoder_steps :]),
    #            time_series_to_array(y.future_flags[..., i][..., None]),
    #        )
    #        y_i = preprocessor.restore_timestamps(past_df) + preprocessor.restore_timestamps(future_df)
    #        x_i = (
    #            preprocessor.preprocessor["target"][entity]
    #            .transform(
    #                np.concatenate([past_df["power_usage"].to_numpy(), future_df["power_usage"].to_numpy()]).reshape(
    #                    (-1, 1)
    #                )
    #            )
    #            .reshape(-1)
    #        )
    #        y_i_truncated = []
    #        x_i_truncated = []
    #
    #        for ii, jj in zip(x_i, y_i):
    #            if truncate_past is None or jj >= truncate_past:
    #                x_i_truncated.append(ii)
    #                y_i_truncated.append(jj)
    #
    #        plt.plot(y_i_truncated, x_i_truncated)
    #
    #    static_df = preprocessor.inverse_transform(
    #        time_series_to_array(x),
    #        time_series_to_array(
    #            np.concatenate([y.static_flags[:, None] for _ in range(self.total_time_steps)], axis=1)
    #        ),
    #    )
    #    static_y = preprocessor.restore_timestamps(static_df)
    #    static_x = (
    #        preprocessor.preprocessor["target"][entity]
    #        .transform(static_df["power_usage"].to_numpy().reshape((-1, 1)))
    #        .reshape(-1)
    #        .reshape(-1),
    #    )
    #    static_x_truncated = []
    #    static_y_truncated = []
    #    for ii, jj in zip(static_x, static_y):
    #        if truncate_past is None or jj >= truncate_past:
    #            static_x_truncated.append(ii)
    #            static_y_truncated.append(jj)
    #
    #    plt.plot(static_y_truncated, static_x_truncated)
    #    plt.legend([i.title() for i in _REAL_INPUTS + _CATEGORICAL_INPUTS + ["Id"]])
    #    plt.title(f"{entity} feature importance".title())
    #    plt.xticks(rotation=90, fontsize=12)
    #    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m.%d %H"))
    #    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=20))
    #    return fig


class ElectricityPreprocessor(Preprocessor):
    def adapt(self, df: pl.DataFrame) -> None:
        total = (count_groups(df, "id") * 2) + len(_CATEGORICAL_INPUTS) + 1

        with tqdm(total=total, desc="Training preprocessor ...") as pbar:
            self.state["categorical"]["id"].adapt(df["id"].to_numpy())
            pbar.update(1)

            for i, sub_df in df.groupby("id"):
                self.state["target"][i].adapt(df[_TARGETS].to_numpy(order="c"))
                self.state["real"][i].adapt(df[_REAL_INPUTS].to_numpy(order="c"))
                pbar.update(2)

            for i in _CATEGORICAL_INPUTS:
                self.state["categorical"][i].adapt(df[i].to_numpy())
                pbar.update(1)
        self.built = True

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        lf_list = []

        total = (count_groups(df, "id") * 2) + len(_CATEGORICAL_INPUTS) + 1

        with tqdm(total=total, desc="Applying preprocessor...") as pbar:
            for df_id, sub_df in df.groupby("id"):
                sub_df: pl.DataFrame
                sub_lf: pl.LazyFrame = sub_df.lazy()

                x_real = sub_df[_REAL_INPUTS].to_numpy(order="c")
                x_real = self.transform_one(x_real, "real", df_id)

                sub_lf = sub_lf.with_columns(
                    [pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_real.T, _REAL_INPUTS)]
                )
                pbar.update(1)

                x_target = sub_df[_TARGETS].to_numpy(order="c")
                x_target = self.transform_one(x_target, "target", df_id)
                sub_lf = sub_lf.with_columns(
                    [pl.lit(i).alias(j).cast(pl.Float32) for i, j in zip(x_target.T, _TARGETS)]
                )
                pbar.update(1)

                lf_list.append(sub_lf)

            df: pl.DataFrame = pl.concat(lf_list).collect()

            for i in _CATEGORICAL_INPUTS:
                x = df[i].to_numpy()
                x = self.transform_one(x, "categorical", i)
                df = df.drop(i).with_columns(pl.lit(x).alias(i).cast(pl.Int8))
                pbar.update(1)

            ids = self.transform_one(df["id"].to_numpy(), "categorical", "id")
            df = df.drop("id").with_columns(id=pl.lit(ids).cast(pl.UInt16)).shrink_to_fit(in_place=True).rechunk()
            pbar.update(1)

        df = df.shrink_to_fit(in_place=True).rechunk()
        return df

    @staticmethod
    def from_dataframe(df: pl.DataFrame) -> ElectricityPreprocessor:
        ids = df["id"].unique().to_list()
        state = {
            "target": {i: layers.Normalization() for i in ids},
            "real": {i: layers.Normalization() for i in ids},
            "categorical": {i: layers.IntegerLookup() for i in _CATEGORICAL_INPUTS},
        }
        state["categorical"]["id"] = layers.StringLookup()
        return ElectricityPreprocessor(state)


# -------------- actual implementations ----------------------------------------


def convert_to_parquet(data_dir: str, output_dir: str | None = None, delete_processed: bool = True):
    if output_dir is None:
        output_dir = data_dir

    with open(f"{data_dir}/LD2011_2014.txt", "r") as file:
        txt_content = file.read()

    csv_content = txt_content.replace(",", ".").replace(";", ",")

    with TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/LD2011_2014.csv", "w+") as file:
            file.write(csv_content)

        pl.scan_csv(f"{tmpdir}/LD2011_2014.csv", infer_schema_length=999999, try_parse_dates=True).rename(
            {"": "timestamp"}
        ).sink_parquet(f"{output_dir}/LD2011_2014.parquet")

    if delete_processed:
        os.remove(f"{data_dir}/LD2011_2014.txt")


def read_parquet(data_dir: str, cutoff_days: Tuple[datetime, datetime]) -> pl.DataFrame:
    lf = pl.scan_parquet(f"{data_dir}/LD2011_2014.parquet")

    num_cols = lf.columns[1:]
    lf = lf.sort("timestamp")
    # down sample to 1h https://pola-rs.github.io/polars-book/user-guide/transformations/time-series/rolling/
    lf = lf.groupby_dynamic("timestamp", every="1h").agg([pl.col(i).mean() for i in num_cols])

    df_list = []

    for label in tqdm(num_cols, desc="Formatting inputs"):
        sub_df: pl.DataFrame = lf.select("timestamp", label)
        lazy_sub_df = sub_df.lazy()
        lazy_sub_df = (
            lazy_sub_df.filter(pl.col("timestamp") >= cutoff_days[0])
            .filter(pl.col("timestamp") <= cutoff_days[1])
            .rename({label: "power_usage"})
            .with_columns(
                [
                    pl.col("power_usage").cast(pl.Float32),
                    pl.col("timestamp").dt.year().alias("year").cast(pl.UInt16),
                    pl.col("timestamp").dt.month().alias("month").cast(pl.UInt8),
                    pl.col("timestamp").dt.hour().alias("hour").cast(pl.UInt8),
                    pl.col("timestamp").dt.day().alias("day").cast(pl.UInt8),
                    pl.col("timestamp").dt.weekday().alias("day_of_week").cast(pl.UInt8),
                ],
                id=pl.lit(label),
            )
        )
        sub_df = lazy_sub_df.collect()
        sub_df = sub_df.shrink_to_fit(in_place=True).rechunk()

        df_list.append(sub_df)

    df: pl.DataFrame = pl.concat(df_list)
    df = df.shrink_to_fit(in_place=True).rechunk()
    log.info(f"{df.null_count() = }")
    return df


def split_data(
    df: pl.DataFrame,
    validation_boundary: datetime = datetime(2014, 8, 8),
    test_boundary: int = datetime(2014, 9, 1),
    split_overlap_days: int = 7,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    This dataset was recorded in interval [2011-01-01, 2015-01-01].
    """

    train_df = df.filter(pl.col("timestamp") < validation_boundary)
    validation_df: pl.DataFrame = (
        df.lazy()
        .filter(pl.col("timestamp") >= validation_boundary - timedelta(days=split_overlap_days))
        .filter(pl.col("timestamp") < test_boundary)
        .collect()
    )
    test_df = df.filter(pl.col("timestamp") >= test_boundary - timedelta(days=split_overlap_days))
    return (
        train_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        validation_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
        test_df.drop("timestamp").shrink_to_fit(in_place=True).rechunk(),
    )


# -------------------- inference ---------------


# def inverse_transform(
#    preprocessor: PreprocessorDict,
#    x_batch: np.ndarray,
#    y_batch: np.ndarray,
# ) -> pl.DataFrame:
#    ids = np.asarray(x_batch[..., 5], dtype=np.int32)
#
#    ids_str = preprocessor["categorical"]["id"].inverse_transform(ids)
#
#    if len(np.unique(ids)) == 1:
#        return inverse_transform_for_single_id(preprocessor, x_batch, y_batch, ids_str[0])
#
#    lf_list = []
#
#    for id_i in tqdm(ids_str):
#        idx_i = np.argwhere(ids == id_i).reshape(-1)
#        x_i = np.take(x_batch, idx_i, axis=0)
#        y_i = np.take(y_batch, idx_i, axis=0)
#
#        df = inverse_transform_for_single_id(preprocessor, x_i, y_i, id_i)
#
#        lf_list.append(df.lazy())
#
#    return pl.concat(lf_list).collect()


# def inverse_transform_for_single_id(
#    preprocessor: PreprocessorDict,
#    x_batch: np.ndarray,
#    y_batch: np.ndarray,
#    entity_id: str,
# ) -> pl.LazyFrame:
#    config: DataConfig = fixed_parameters.get_config("electricity")
#
#    y_new = preprocessor["target"][entity_id].inverse_transform(y_batch).reshape(-1)
#
#    x_categorical = np.take(x_batch, list(config.input_known_categorical_idx) + list(config.input_static_idx), axis=-1)
#
#    x_categorical_new = [
#        preprocessor["categorical"][j].inverse_transform(np.asarray(i, dtype=np.int32))
#        for i, j in zip(x_categorical.T, _CATEGORICAL_INPUTS + [_ID_COLUMN])
#    ]
#
#    x_real = np.take(x_batch, list(config.input_known_real_idx) + list(config.input_observed_idx), axis=-1)
#
#    x_real_new = preprocessor["real"][entity_id].inverse_transform(x_real).reshape(-1)
#
#    return (
#        pl.DataFrame()
#        .with_columns([pl.lit(i).alias(j) for i, j in zip(x_categorical_new, _CATEGORICAL_INPUTS + [_ID_COLUMN])])
#        .with_columns([pl.lit(x_real_new).alias("year").cast(pl.UInt16), pl.lit(y_new).alias("power_usage")])
#        .rechunk()
#    )


# def restore_timestamps(df: pl.DataFrame) -> List[datetime]:
#    return df.with_columns(
#        [
#            pl.datetime("year", "month", "day", "hour")
#            .dt.strftime("%Y/%m/%d %H:%M:%S")
#            .alias("timestamp")
#            .str.to_datetime()
#        ]
#    )["timestamp"].to_list()
