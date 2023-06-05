from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import auto, IntEnum
from typing import (
    NamedTuple,
    List,
    Dict,
    Tuple,
    Sequence,
    ClassVar,
    Mapping,
    TypeVar,
    Hashable,
    Callable,
    DefaultDict,
)
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from keras_pbar import keras_pbar

# from keras.utils.tf_utils import can_jit_compile
from sklearn.preprocessing import StandardScaler, LabelEncoder

# from jax.tree_util import tree_map, tree_map_with_path
# from temporal_fusion_transformer.modeling import TFTInputs
# class DataEntry(tf.experimental.BatchableExtensionType):
#     inputs: TFTInputs
#     outputs: tf.Tensor
#     time: tf.Tensor
#     id: tf.Tensor


class DataTypes(IntEnum):
    REAL_VALUED = auto()
    CATEGORICAL = auto()
    DATE = auto()


class InputTypes(IntEnum):
    TARGET = auto()
    OBSERVED_INPUT = auto()
    KNOWN_INPUT = auto()
    STATIC_INPUT = auto()
    # Single column used as an entity identifier
    ID = auto()
    # Single column exclusively used as a time index
    TIME = auto()


class SchemaEntry(NamedTuple):
    data_type: DataTypes
    input_type: InputTypes


class DataParams(NamedTuple):
    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    num_outputs: int


class ModelParams(NamedTuple):
    dropout_rate: float
    hidden_layer_size: int
    num_attention_heads: int
    max_gradient_norm: float
    num_attention_heads: int


class Experiment(ABC):

    """
    Attributes:

    name:
        Name of the experiment, obviously.
    data_params:
        Data parameters are the one defined by dataset properties, they are static.
    model_params:
        Model parameters are subject to hyperparameter optimization.
    column_schema:

    """

    name: ClassVar[str]
    column_schema: ClassVar[Dict[str, SchemaEntry | List[SchemaEntry]]]
    model_params: ClassVar[ModelParams]
    data_params: ClassVar[DataParams]

    # @abstractmethod
    # def train_test_split(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    #    raise NotImplementedError

    @classmethod
    def get_single_col_by_input_type(cls, input_type: InputTypes) -> str:
        """
        Returns name of single column.

        Parameters
        ----------
        input_type:
            Input type of column to extract

        Returns
        -------
        """
        col = cls.get_cols_by_input_type(input_type)

        if len(col) != 1:
            raise ValueError("Invalid number of columns for {}".format(input_type))

        return col[0]

    @classmethod
    def get_cols_by_input_type(
        cls,
        input_type: InputTypes,
        excluded_data_types: Sequence[DataTypes] | None = None,
    ) -> List[str]:
        if excluded_data_types is None:
            excluded_data_types = []

        def filter_func(i: SchemaEntry | List[SchemaEntry]) -> bool:
            if isinstance(i, List):
                return any(map(filter_func, i))
            return i.input_type == input_type and i.data_type not in excluded_data_types

        return list(filter_dict(cls.column_schema, value_filter=filter_func).keys())

    @classmethod
    def get_cols_by_data_type(
        cls,
        data_type: DataTypes,
        excluded_input_types: Sequence[InputTypes] | None = None,
    ) -> List[str]:
        """
        Extracts the names of columns that correspond to a define data_type.


        Parameters
        ----------
        data_type:
            DataType of columns to extract.
        excluded_input_types:
            Set of input types to exclude

        Returns
        -------

        """
        if excluded_input_types is None:
            excluded_input_types = []

        def filter_func(i: SchemaEntry | List[SchemaEntry]) -> bool:
            if isinstance(i, List):
                return any(map(filter_func, i))
            return i.input_type not in excluded_input_types and i.data_type == data_type

        return list(filter_dict(cls.column_schema, value_filter=filter_func).keys())


class ElectricityExperiment(Experiment):
    """
    Electricity is a widely used dataset described by M. Harries and analyzed by J. Gama (see papers below).
    This data was collected from the Australian New South Wales Electricity Market.  In this market, prices are not
    fixed and are affected by demand and supply of the market. They are set every five minutes. Electricity transfers
    to/from the neighboring state of Victoria were done to alleviate fluctuations. The dataset (originally named ELEC2)
    contains 45,312 instances dated from 7 May 1996 to 5 December 1998. Each example of the dataset refers to a period
    of 30 minutes, i.e. there are 48 instances for each time period of one day. Each example on the dataset has
    5 fields, the day of week, the time stamp, the New South Wales electricity demand, the Victoria electricity demand,
    the scheduled electricity transfer between states and the class label. The class label identifies the change of the
    price (UP or DOWN) in New South Wales relative to a moving average of the last 24 hours (and removes the impact of
    longer term price trends).


    References:
    ----------
    .. [1] https://www.openml.org/d/151

    """

    name: ClassVar[str] = "electricity"
    total_time_steps = 8 * 24
    test_boundary: ClassVar[int] = 1339
    data_params: ClassVar[DataParams] = DataParams(
        num_encoder_steps=7 * 24,
        num_outputs=1,
        known_categories_sizes=[],
        static_categories_sizes=[369],
    )
    model_params: ClassVar[ModelParams] = ModelParams(
        dropout_rate=0.1,
        hidden_layer_size=160,
        max_gradient_norm=1e-2,
        num_attention_heads=4,
    )
    column_schema: ClassVar[Dict[str, SchemaEntry | List[SchemaEntry]]] = {
        "id": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.ID),
        "hours_from_start": [
            SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TIME),
            SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ],
        "power_usage": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TARGET),
        "hour": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        "day_of_week": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        "categorical_id": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    }

    @classmethod
    def from_raw_csv(
        cls, csv_path: str, validation_boundary: int = 1315
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Read raw CSV, pre-process it, create TF dataset. You probably will want to execute it only once,
        and then write dataset to local filesystem (or mb even create squshFS image).

        Parameters
        ----------
        csv_path:
            Path to raw unprocessed CSV file.
        validation_boundary:
            Year at which validation data should start.

        Returns
        -------

        experiments:
            Instance of experiment, with train and validation splits loaded. Dataset yields elements of shape
            {
                "identifier": ???,
                "time": ???,
                "outputs": ???,
                "inputs_static": ???,
                "inputs_known_categorical": ???,
                "inputs_known_real": ???,
                "inputs_observed": ???
            }
            The dataset is not batched, and must not be further shuffled.


        Examples
        -------

        >>> from temporal_fusion_transformer.experiments import ElectricityExperiment
        >>> experiment = ElectricityExperiment.from_raw_csv("data.csv")
        >>> train_ds, val_ds = experiment.train_test_split()
        >>> train_ds.save("data/train")
        >>> val_ds.save("data/validation")
        """
        logging.info(f"Loading electricity dataset from {csv_path}")
        # This code was copy pasted from original implementation, and I have very little idea
        # what is it doing. TODO: rewrite using polars (and figure out WTF is going on here).
        df = pd.read_csv(csv_path, index_col=0, sep=";", decimal=",")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Used to determine the start and end dates of a series
        df = df.resample("1h").mean().replace(0.0, np.nan)

        earliest_time = df.index.min()

        df_list = []

        for label in keras_pbar(df):
            srs = df[label]

            start_date = min(srs.fillna(method="ffill").dropna().index)
            end_date = max(srs.fillna(method="bfill").dropna().index)

            active_range = (srs.index >= start_date) & (srs.index <= end_date)
            srs = srs[active_range].fillna(0.0)

            tmp = pd.DataFrame({"power_usage": srs})
            date = tmp.index
            tmp["t"] = (date - earliest_time).seconds / 60 / 60 + (
                date - earliest_time
            ).days * 24
            tmp["days_from_start"] = (date - earliest_time).days
            tmp["categorical_id"] = label
            tmp["date"] = date
            tmp["id"] = label
            tmp["hour"] = date.hour
            tmp["day"] = date.day
            tmp["day_of_week"] = date.dayofweek
            tmp["month"] = date.month

            df_list.append(tmp)

        df = pd.concat(df_list, axis=0, join="outer").reset_index(drop=True)
        del df_list

        df["categorical_id"] = df["id"].copy()
        df["hours_from_start"] = df["t"]
        df["categorical_day_of_week"] = df["day_of_week"].copy()
        df["categorical_hour"] = df["hour"].copy()

        # Filter to match range used by other academic papers
        df = df[(df["days_from_start"] >= 1096) & (df["days_from_start"] < 1346)].copy()
        logging.info("Done.")

        # index: days_from_start, Length: 2198072, dtype: int64
        # np.max(index)
        # Out[43]: 1345
        # np.min(index)
        # Out[44]: 1096

        index = df["days_from_start"]
        train_df = df.loc[index < validation_boundary]
        valid_df = df.loc[(index >= validation_boundary - 7)]

        # Pre-processing will do few things:
        # - Find real columns and apply sklearn.NormalScaler to them instance-wise (using ID column).
        # - Find target columns and apply sklearn.NormalScaler to them instance-wise (using ID column).
        # - Find categorical columns and apply sklearn.LabelEncoder category-wise.
        # - Sample data grouped by ID and sorted by TIME.
        # - Group Data into ExperimentDataBatch instances.
        # - Create train-test split.
        # - Convert to TF dataset.
        # It expects data frame with following schema:
        # power_usage  ...  categorical_day_of_week  categorical_hour
        # 0  17544     2.538071  ...                        2                 0
        # 1  17545     2.855330  ...                        2                 1
        # 2  17546     2.855330  ...                        2                 2
        # 3  17547     2.855330  ...                        2                 3
        # 4  17548     2.538071  ...                        2                 4
        # ----------- collect all column types ---------------------
        id_column = cls.get_single_col_by_input_type(InputTypes.ID)
        target_column = cls.get_single_col_by_input_type(InputTypes.TARGET)
        # Format real scales
        real_inputs = cls.get_cols_by_data_type(
            DataTypes.REAL_VALUED, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = cls.get_cols_by_data_type(
            DataTypes.CATEGORICAL, {InputTypes.ID, InputTypes.TIME}
        )
        time_col = cls.get_single_col_by_input_type(InputTypes.TIME)
        input_static_cols = cls.get_cols_by_input_type(InputTypes.STATIC_INPUT)
        input_observed = cls.get_cols_by_input_type(InputTypes.OBSERVED_INPUT)
        input_known_real = cls.get_cols_by_input_type(
            InputTypes.KNOWN_INPUT, {DataTypes.CATEGORICAL, DataTypes.DATE}
        )
        input_known_categorical = cls.get_cols_by_input_type(
            InputTypes.KNOWN_INPUT, {DataTypes.REAL_VALUED, DataTypes.DATE}
        )

        # Initialize scalers/label encoders.
        real_scalers: Dict[str, StandardScaler] = {}
        target_scaler: Dict[str, StandardScaler] = {}
        categorical_scalers: Dict[str, LabelEncoder] = {}
        num_classes = {}
        logging.debug("Fitting scalers.")

        for identifier, sliced in keras_pbar(df.groupby(id_column)):
            if len(sliced) >= cls.total_time_steps:
                # Fit scalers.
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                real_scalers[identifier] = StandardScaler().fit(data)
                target_scaler[identifier] = StandardScaler().fit(targets)

        for col in keras_pbar(categorical_inputs):
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            num_classes[col] = srs.nunique()
            categorical_scalers[col] = LabelEncoder().fit(srs.values)

        logging.debug(f"{num_classes = }")

        def apply_scalers(_df: pd.DataFrame) -> pd.DataFrame:
            _df_list = []
            for _identifier, _sliced in keras_pbar(_df.groupby(id_column)):
                if len(_sliced) >= cls.total_time_steps:
                    sliced_copy = _sliced.copy()
                    sliced_copy[real_inputs] = real_scalers[_identifier].transform(
                        sliced_copy[real_inputs].values
                    )
                    _df_list.append(sliced_copy)

            for _col in keras_pbar(categorical_inputs):
                # Set all to str so that we don't have mixed integer/string columns
                _srs = _df[_col].apply(str)
                _df[_col] = categorical_scalers[_col].transform(_srs)
                _df = pd.concat(_df_list, axis=0)

            return _df

        train_df = apply_scalers(train_df)
        valid_df = apply_scalers(valid_df)

        col_mappings = {
            "identifier": [id_column],
            "time": [time_col],
            "outputs": [target_column],
            "inputs_static": input_static_cols,
            "inputs_known_categorical": input_known_categorical,
            "inputs_known_real": input_known_real,
            "inputs_observed": input_observed,
        }

        def make_np_araay_dict(_df: pd.DataFrame) -> Dict[str, np.ndarray]:
            logging.debug("Grouping and batching data.")
            _df.sort_values(by=[id_column, time_col], inplace=True)
            data_map: DefaultDict[str, List[np.ndarray]] = defaultdict(lambda: [])

            for _, _sliced in keras_pbar(df.groupby(id_column)):
                for k in col_mappings:
                    cols = col_mappings[k]
                    if len(cols) == 0:
                        continue
                    arr = batch_single_entity(
                        _sliced[cols].copy(), cls.total_time_steps
                    )
                    data_map[k].append(arr)

            data_map = dict(**data_map)
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)
            data_map["outputs"] = data_map["outputs"][
                :, cls.data_params.num_encoder_steps :, :
            ]
            return data_map

        train_ds = make_np_araay_dict(train_df)
        val_ds = make_np_araay_dict(valid_df)
        return train_ds, val_ds

    def __init__(
        self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset
    ):
        """Do not use directly! Use from_raw_csv or from_dataframe instead."""
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset


T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable, covariant=True)


def map_dict(
    dictionary: Mapping[K, T],
    value_mapper: Callable[[T], R] | None = None,
    key_mapper: Callable[[K], R] | None = None,
) -> Dict[K | R, V | R]:
    """Applies func to values in dict. Additionally, if provided can also map keys."""

    def identity(x: T) -> T:
        return x

    if value_mapper is None:
        value_mapper = identity

    if key_mapper is None:
        key_mapper = identity

    result = {}
    for k, v in dictionary.items():
        result[key_mapper(k)] = value_mapper(v)
    return result


def filter_dict(
    dictionary: Mapping[K, V],
    key_filter: Callable[[K], bool] | None = None,
    value_filter: Callable[[V], bool] | None = None,
) -> Dict[K, V]:
    def tautology(arg) -> bool:
        # Tautology is an expression, which is always true.
        return True

    if key_filter is None:
        key_filter = tautology

    if value_filter is None:
        value_filter = tautology

    result = {}

    for k, v in dictionary.items():
        if key_filter(k) and value_filter(v):
            result[k] = v

    return result


def batch_single_entity(input_data: pd.Series, lags: int) -> np.ndarray:
    time_steps = len(input_data)
    x = input_data.values
    if time_steps >= lags:
        return np.stack(
            [x[i : time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1
        )
    logging.error("time_steps < lags, this is not expected.")


# @tf.function(
#     reduce_retracing=True,
#     jit_compile=can_jit_compile(),
#     experimental_autograph_options=tf.autograph.experimental.Feature.ALL
# )
# def tf_batch_single_entity(input_data: tf.Tensor, lags: int) -> tf.Tensor:
#     time_steps = tf.shape(input_data)[0]
#     arr = tf.TensorArray(
#         size=lags,
#         dtype=input_data.dtype,
#         clear_after_read=True
#     )
#     for i in tf.range(lags):
#         indexes = tf.range(i, time_steps - (lags - 1) + i, dtype=tf.int32)
#         x = tf.gather(input_data, indexes, axis=0,)
#         arr = arr.write(i, x)
#
#     arr = arr.stack()
#     arr = tf.transpose(arr, [1, 0, 2])
#     return arr

# def make_data_entry(xy: Mapping[str, tf.Tensor]) -> DataEntry:
#     def make_extension_type(x: Mapping[str, tf.Tensor]) -> TFTInputs:
#         return TFTInputs(
#             static=x["inputs_static"],
#             known_real=x.get("inputs_known_real"),
#             known_categorical=x.get("inputs_known_categorical"),
#             observed=x.get("inputs_observed"),
#         )
#
#     return DataEntry(
#         inputs=make_extension_type(xy),
#         outputs=xy["outputs"],
#         time=xy["time"],
#         id=xy["identifier"],
#     )


def concatenate_datasets(
    ds1: tf.data.Dataset | None, ds2: tf.data.Dataset
) -> tf.data.Dataset:
    if ds1 is None:
        return ds2
    return ds1.concatenate(ds2)
