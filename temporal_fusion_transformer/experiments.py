from __future__ import annotations

import functools
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
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from keras_pbar import keras_pbar
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import gen_batches
from absl_extra.collection_utils import map_dict, filter_dict

if TYPE_CHECKING:
    pass


class classproperty(property):
    def __init__(self, getter):
        super().__init__()
        self.getter = getter

    def __get__(self, instance, owner):  # noqa
        return self.getter(owner)


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
    hidden_layer_size: int
    num_attention_heads: int
    dropout_rate: float = 0.1


class OptimizerParams(NamedTuple):
    learning_rate: float
    max_gradient_norm: float


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

    @classproperty
    @abstractmethod
    def column_schema(self) -> Dict[str, SchemaEntry | List[SchemaEntry]]:
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def default_params(self) -> Tuple[ModelParams, OptimizerParams]:
        """
        Model parameters, are the ones, which are model architecture parameters, which are subject to hyperparameter
        fine-tuning, e.g, number of attention heads. Same goes for OptimizerParams.
        Refer to  Deep Learning Tuning Playbook for step-by-step guide. For solved experiments,
        this property will already contain the best ones.

        References
        -------
        .. [1] https://github.com/google-research/tuning_playbook

        Returns
        -------

        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def fixed_params(self) -> DataParams:
        """
        Fixed parameters, are the ones, caused by underlying datasets structure.
        E.g., number of static inputs.

        Returns
        -------

        retval:
            DataParams instance.

        """
        raise NotImplementedError

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

    total_time_steps: ClassVar[int] = 8 * 24
    test_boundary: ClassVar[int] = 1339

    @classproperty
    def column_schema(self) -> Dict[str, SchemaEntry | List[SchemaEntry]]:
        return {
            "id": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.ID),
            "hours_from_start": [
                SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TIME),
                SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ],
            "power_usage": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TARGET),
            "hour": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            "day_of_week": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            "categorical_id": SchemaEntry(
                DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT
            ),
        }

    @classproperty
    def default_params(self) -> Tuple[ModelParams, OptimizerParams]:
        return (
            ModelParams(hidden_layer_size=160, num_attention_heads=4),
            OptimizerParams(learning_rate=1e-3, max_gradient_norm=1e-2),
        )

    @classproperty
    def fixed_params(self) -> DataParams:
        return DataParams(
            num_encoder_steps=7 * 24,
            num_outputs=1,
            known_categories_sizes=[],
            static_categories_sizes=[369],
        )

    @classproperty
    def num_encoder_steps(self) -> int:
        return self.fixed_params.num_encoder_steps

    @classmethod
    def from_raw_csv(
        cls, csv_path: str, validation_boundary: int = 1315, test_boundary=1339
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Read raw CSV, pre-process it, create TF dataset. You probably will want to execute it only once,
        and then write dataset to local filesystem (or mb even create squshFS image).

        Parameters
        ----------
        csv_path:
            Path to raw unprocessed CSV file.
        validation_boundary:
            Year at which validation data should start.
        test_boundary:
            Year at which test data should start.

        Returns
        -------

        experiments:
            Instance of experiment, with train and validation splits loaded. Dataset yields elements of shape
            {
                "identifier": (batch, total_time_steps, 1),
                "time": (batch, total_time_steps, 1),
                "outputs": (batch, total_time_steps, 1),
                "inputs_static": (batch, total_time_steps, 1),
                "inputs_known_real": (batch, total_time_steps, 3),
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
        # what is it doing. TODO: rewrite using polars.
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
        validation_df = df.loc[
            (index >= validation_boundary - 7) & (index < test_boundary)
        ]
        test_df = df.loc[index >= test_boundary - 7]

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
        target_scalers: Dict[str, StandardScaler] = {}
        categorical_scalers: Dict[str, LabelEncoder] = {}
        num_classes = {}
        logging.debug("Fitting scalers.")
        for identifier, sliced in keras_pbar(df.groupby(id_column)):
            if len(sliced) >= cls.total_time_steps:
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                real_scalers[identifier] = StandardScaler().fit(data)
                target_scalers[identifier] = StandardScaler().fit(targets)

        for col in keras_pbar(categorical_inputs):
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            num_classes[col] = srs.nunique()
            categorical_scalers[col] = LabelEncoder().fit(srs.values)

        logging.debug(f"{num_classes = }")
        apply_fn = functools.partial(
            normalize_data,
            id_column=id_column,
            categorical_inputs=categorical_inputs,
            categorical_scalers=categorical_scalers,
            real_scalers=real_scalers,
            real_inputs=real_inputs,
            total_time_steps=cls.total_time_steps,
        )

        train_df = apply_fn(train_df)
        validation_df = apply_fn(validation_df)
        test_df = apply_fn(test_df)

        col_mapping = {
            "identifier": [id_column],
            "time": [time_col],
            "outputs": [target_column],
            "inputs_static": input_static_cols,
            "inputs_known_categorical": input_known_categorical,
            "inputs_known_real": input_known_real,
            "inputs_observed": input_observed,
        }

        apply_fn = functools.partial(
            make_np_array_dict,
            id_column=id_column,
            time_col=time_col,
            col_mapping=col_mapping,
            total_time_steps=cls.total_time_steps,
            num_encoder_steps=cls.num_encoder_steps,
        )

        train_ds = apply_fn(train_df)
        validation_ds = apply_fn(validation_df)
        test_ds = apply_fn(test_df)
        return train_ds, validation_ds, test_ds


T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable, covariant=True)


def batch_single_entity(input_data: pd.Series, lags: int) -> np.ndarray:
    time_steps = len(input_data)
    x = input_data.values
    if time_steps >= lags:
        return np.stack(
            [x[i : time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1
        )
    logging.error("time_steps < lags, this is not expected.")


def export_sharded_dataset(
    data: Mapping[str, np.ndarray], export_path: str, shard_size: int = 100_000
):
    """
    Split dataset in shards of size `shard_size`, and write the as TF protobuf to local file system.

    Parameters
    ----------
    data
    export_path
    shard_size

    Returns
    -------

    """
    n = len(data["identifier"])
    batches = gen_batches(n, shard_size)

    n_batches = n // shard_size
    if n % shard_size != 0:
        n_batches += 1

    for index, shard_slice in keras_pbar(enumerate(batches), n_batches):
        shard = map_dict(
            data, value_mapper=lambda v: v[shard_slice.start : shard_slice.stop]
        )
        tf.data.Dataset.from_tensors(shard).save(f"{export_path}/{index}")


def normalize_data(
    df: pd.DataFrame,
    *,
    id_column: str,
    total_time_steps: int,
    real_inputs: Sequence[str],
    real_scalers: Mapping[str, StandardScaler],
    categorical_scalers: Mapping[str, LabelEncoder],
    categorical_inputs: Sequence[str],
) -> pd.DataFrame:
    df_list = []
    for identifier, sliced in keras_pbar(df.groupby(id_column)):
        if len(sliced) >= total_time_steps:
            sliced_copy = sliced.copy()
            sliced_copy[real_inputs] = real_scalers[identifier].transform(
                sliced_copy[real_inputs].values
            )
            df_list.append(sliced_copy)

    df = pd.concat(df_list, axis=0)
    for col in keras_pbar(categorical_inputs):
        string_df = df[col].apply(str)
        df[col] = categorical_scalers[col].transform(string_df)

    return df


def make_np_array_dict(
    df: pd.DataFrame,
    *,
    id_column: str,
    time_col: str,
    col_mapping: Mapping[str, str | Sequence[str]],
    total_time_steps: int,
    num_encoder_steps: int,
) -> Dict[str, np.ndarray]:
    logging.debug("Grouping and batching data.")
    df.sort_values(by=[id_column, time_col], inplace=True)
    data_map: DefaultDict[str, List[np.ndarray]] = defaultdict(lambda: [])

    for _, sliced in keras_pbar(df.groupby(id_column)):
        for k in col_mapping:
            cols = col_mapping[k]
            if len(cols) == 0:
                continue
            arr = batch_single_entity(sliced[cols].copy(), total_time_steps)
            if arr.dtype == np.int64:
                arr = arr.astype(np.int32)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            data_map[k].append(arr)

    data_map = dict(**data_map)
    for k in data_map:
        data_map[k] = np.concatenate(data_map[k], axis=0)
    # Save only future steps
    data_map["outputs"] = data_map["outputs"][:, num_encoder_steps:]
    # Static are not time varying
    data_map["inputs_static"] = data_map["inputs_static"][:, 0]
    return data_map
