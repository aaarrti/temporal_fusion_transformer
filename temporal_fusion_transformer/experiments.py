from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
)
from collections import defaultdict

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from temporal_fusion_transformer.modeling import TFTInputs


class DataEntry(tf.experimental.BatchableExtensionType):
    inputs: TFTInputs
    outputs: tf.Tensor
    time: tf.Tensor
    identifiers: tf.Tensor


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
    """
    Attributes
    ----------

    static_categories_sizes:
    known_categories_sizes:
    num_encoder_steps:
    num_outputs:

    """

    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    num_outputs: int


class ModelParams(NamedTuple):
    """
    Attributes
    ----------

    dropout_rate:
    hidden_layer_size:
    num_attention_heads:
    num_heads:
    max_graqient_norm":


    """

    learning_rate: float = 1e-3
    dropout_rate: float = 0.1
    hidden_layer_size: int = 5
    num_attention_heads: int = 4
    max_gradient_norm: float = 0.01
    num_heads = 4


class Experiment(ABC):
    @property
    @abstractmethod
    def fixed_params(self) -> DataParams:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_hyperparams(self) -> ModelParams:
        pass

    @property
    @abstractmethod
    def train_test_split(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Returns
        -------

        train_dataset:
            Dataset, which in each iteration yields instance of ExperimentDataBatch.
        validation_dataset:
            Dataset, which in each iteration yields instance of ExperimentDataBatch.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


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
    .. [1] M. Harries, Splice-2 comparative evaluation: Electricity pricing.
        Technical report, The University of South Wales, 1999.
    .. [2] J. Gama, P. Medas, G. Castillo, and P. Rodrigues. Learning with drift detection.
        In SBIA Brazilian Symposium on Artificial Intelligence, pages 286–295, 2004.
    .. [3] https://www.openml.org/d/151

    """

    total_time_steps: ClassVar[int] = 8 * 24
    num_encoder_steps: ClassVar[int] = 7 * 24
    num_outputs: ClassVar[int] = 1
    test_boundary = 1339
    column_schema: ClassVar[Dict[str, SchemaEntry]] = {
        "id": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.ID),
        "hours_from_start": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TIME),
        "power_usage": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TARGET),
        "hour": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        "day_of_week": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        "categorical_id": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    }

    def __init__(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds

    @classmethod
    def from_raw_csv(cls, csv_path: str) -> ElectricityExperiment:
        logging.info(f"Loading electricity dataset from {csv_path}")
        # This code was copy pasted from original implementation, and I have very little idea
        # what is it doing.
        df = pd.read_csv(csv_path, index_col=0, sep=";", decimal=",")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Used to determine the start and end dates of a series
        output = df.resample("1h").mean().replace(0.0, np.nan)

        earliest_time = output.index.min()

        df_list = []
        for label in output:
            logging.debug("Processing {}".format(label))
            srs = output[label]

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

        output = pd.concat(df_list, axis=0, join="outer").reset_index(drop=True)

        output["categorical_id"] = output["id"].copy()
        output["hours_from_start"] = output["t"]
        output["categorical_day_of_week"] = output["day_of_week"].copy()
        output["categorical_hour"] = output["hour"].copy()

        # Filter to match range used by other academic papers
        output = output[
            (output["days_from_start"] >= 1096) & (output["days_from_start"] < 1346)
        ].copy()
        logging.info("Done.")

        return cls.from_dataframe(output)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ElectricityExperiment:
        """
        This method accepts raw un-process CSV file for Electricity dataset, and convert it into ready to use
        TensorFlow dataset and save it in instance property. Pre-processing will do few things:
            - Find real columns and apply sklearn.NormalScaler to them instance-wise (using ID column).
            - Find target columns and apply sklearn.NormalScaler to them instance-wise (using ID column).
            - Find categorical columns and apply sklearn.LabelEncoder category-wise.
            - Sample data grouped by ID and sorted by TIME.
            - Group Data into ExperimentDataBatch instances.
            - Create train-test split.
            - Convert to TF dataset.

        It expects data frame with following schema:
            power_usage  ...  categorical_day_of_week  categorical_hour
            0  17544     2.538071  ...                        2                 0
            1  17545     2.855330  ...                        2                 1
            2  17546     2.855330  ...                        2                 2
            3  17547     2.855330  ...                        2                 3
            4  17548     2.538071  ...                        2                 4

        Parameters
        ----------
        df:
            Raw CSV file loaded into pandas.Dataframe.

        Returns
        -------

        retval:
            Experiment Instance.

        """
        id_column = get_single_col_by_input_type(InputTypes.ID, cls.column_schema)
        target_column = get_single_col_by_input_type(
            InputTypes.TARGET, cls.column_schema
        )

        # Format real scales
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, cls.column_schema, {InputTypes.ID, InputTypes.TIME}
        )

        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, cls.column_schema, {InputTypes.ID, InputTypes.TIME}
        )

        real_scalers: Dict[str, StandardScaler] = {}
        # We need it for inverse transform for predictions?
        target_scaler: Dict[str, StandardScaler] = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):
            if len(sliced) >= cls.total_time_steps:
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                real_scalers[identifier] = StandardScaler().fit(data)
                target_scaler[identifier] = StandardScaler().fit(targets)
            identifiers.append(identifier)

        categorical_scalers: Dict[str, LabelEncoder] = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

            # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_column):
            # Filter out any trajectories that are too short
            if len(sliced) >= cls.total_time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values
                )
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = categorical_scalers[col].transform(string_df)

        data = output

        time_col = get_single_col_by_input_type(InputTypes.TIME, cls.column_schema)

        data.sort_values(by=[id_column, time_col], inplace=True)

        logging.debug("Getting valid sampling locations.")
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_column):
            logging.debug("Getting locations for {}".format(identifier))
            num_entries = len(df)
            if num_entries >= cls.total_time_steps:
                valid_sampling_locations += [
                    (identifier, cls.total_time_steps + i)
                    for i in range(num_entries - cls.total_time_steps + 1)
                ]
            split_data_map[identifier] = df

        input_static_cols = list(
            filter_dict(
                cls.column_schema,
                value_filter=lambda v: v.input_type == InputTypes.STATIC_INPUT,
            ).keys()
        )

        input_observed = list(
            filter_dict(
                cls.column_schema,
                value_filter=lambda v: v.input_type == InputTypes.OBSERVED_INPUT,
            ).keys()
        )

        input_known_real = list(
            filter_dict(
                cls.column_schema,
                value_filter=lambda v: (
                    v.input_type == InputTypes.KNOWN_INPUT
                    and v.data_type == DataTypes.REAL_VALUED
                ),
            ).keys()
        )

        input_known_categorical = list(
            filter_dict(
                cls.column_schema,
                value_filter=lambda v: (
                    v.input_type == InputTypes.KNOWN_INPUT
                    and v.data_type == DataTypes.CATEGORICAL
                ),
            ).keys()
        )

        data_map = defaultdict(lambda: [])
        for _, sliced in data.groupby(id_column):
            col_mappings = {
                "identifier": [id_column],
                "time": [time_col],
                "outputs": [target_column],
                "inputs": {
                    "static": input_static_cols,
                    "known_categorical": input_known_categorical,
                    "known_real": input_known_real,
                    "observed": input_observed,
                },
            }

            for k in col_mappings:
                if k == "inputs":
                    for kk in col_mappings[k]:
                        cols = col_mappings[kk]
                        arr = batch_single_entity(
                            sliced[cols].copy(), cls.total_time_steps
                        )
                        data_map[k][kk].append(arr)

                cols = col_mappings[k]
                arr = batch_single_entity(sliced[cols].copy(), cls.total_time_steps)
                data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        # Shorten target so we only get decoder steps
        data_map["outputs"] = data_map["outputs"][:, cls.num_encoder_steps :, :]

        # FIXME plz
        data_entry = DataEntry()

        return data_map

    @property
    def fixed_params(self) -> DataParams:
        return DataParams(
            static_categories_sizes=[],
            known_categories_sizes=[],
            num_encoder_steps=self.num_encoder_steps,
            num_outputs=self.num_outputs,
        )

    @property
    def default_hyperparams(self) -> ModelParams:
        pass

    @property
    def train_test_split(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        return ()


def get_single_col_by_input_type(
    input_type: InputTypes, column_definition: Dict[str, SchemaEntry]
) -> str:
    """
    Returns name of single column.

    Parameters
    ----------
    input_type:
        Input type of column to extract
    column_definition:
        Column definition list for experiment

    Returns
    -------
    """

    def filter_func(i: SchemaEntry):
        return i.input_type == input_type

    col = list(filter_dict(column_definition, value_filter=filter_func).keys())

    if len(col) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return col[0][0]


def extract_cols_from_data_type(
    data_type: DataTypes,
    column_definition: Dict[str, SchemaEntry],
    excluded_input_types: Sequence[InputTypes],
) -> List[str]:
    """
    Extracts the names of columns that correspond to a define data_type.


    Parameters
    ----------
    data_type:
        DataType of columns to extract.
    column_definition:
        Column definition to use.
    excluded_input_types:
        Set of input types to exclude

    Returns
    -------

    """

    def filter_func(i: SchemaEntry):
        return i.input_type not in excluded_input_types and i.data_type == data_type

    return list(filter_dict(column_definition, value_filter=filter_func).keys())


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
    def tautology(arg):
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


def batch_single_entity(input_data: pd.Series, lags: int) -> np.ndarray | None:
    time_steps = len(input_data)
    x = input_data.values
    if time_steps >= lags:
        return np.stack(
            [x[i : time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1
        )

    else:
        return None
