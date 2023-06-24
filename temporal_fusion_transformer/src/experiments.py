from __future__ import annotations

import datetime
import gc
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import auto, IntEnum
from functools import cached_property
from typing import NamedTuple, List, Dict, Sequence, Tuple, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from keras_pbar import keras_pbar
from sklearn.preprocessing import LabelEncoder, StandardScaler

from temporal_fusion_transformer.src.utils import filter_dict

try:
    import cudf  # noqa

    pd = cudf
    logging.info("CuDF will be used for data pre-processing.")
    cudf_available = True
except ModuleNotFoundError:
    logging.info("No CuDF installation found, falling back to pandas processing.")
    cudf_available = False


if TYPE_CHECKING:
    from temporal_fusion_transformer.src.inference import TargetScaler


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


class FixedParams(NamedTuple):
    static_categories_sizes: List[int]
    known_categories_sizes: List[int]
    num_encoder_steps: int
    total_time_steps: int
    num_outputs: int


class Experiment(ABC):
    @property
    @abstractmethod
    def fixed_params(self) -> FixedParams:
        """Fixed parameters, are the ones, caused by underlying datasets structure. E.g., number of static inputs."""
        raise NotImplementedError

    @abstractmethod
    def make_target_scaler(self, save_path) -> TargetScaler:
        """
        Make object, to use dor scaling back the output(s) during inference.
        Target scalers, takes 2 arguments, id of the entity and output.
        """
        raise NotImplementedError

    def process_raw_data(self, data_path: str, save_path: str):
        """
        Suggested data processing follows almost cookbook-like recipe
        - read csv(s)
        - parse data time
        - resample
        - filter/fill NaN's
        - split data
        - train sklearn normal scalers and categorical label encoders
        - apply scalers and label encoders
        - convert each inputs for each entity into time-series
        - convert to TF dataset
        ...
          Read raw CSV at csv_path, pre-process it, create TF dataset and save respective splits to
        - <save_path>/electricity/training
        - <save_path>/electricity/validation
        - <save_path>/electricity/test

        In order to save disc memory, the identifiers and time are not saved for training splits,
        and only outputs corresponding to future timestamps are saved. You would probably still want to create
        SquashFS image from it, e.g., with mksquashfs "data" "data.sqfs" -all-root -action 'chmod(o+rX)@!perm(o+rX)'.

        Parameters
        ----------
        data_path:
        save_path:
        Returns
        -------

        """
        df = self._read_raw_csv(data_path)
        label_encoders = self._fit_label_encoders(df)

        train_df, validation_df, test_df = self._split_data(df)
        real_scalers, target_scalers = self._fit_scalers(train_df)

        train_df = self._normalize_data(train_df, real_scalers)
        validation_df = self._normalize_data(validation_df, real_scalers)
        test_df = self._normalize_data(test_df, real_scalers)

        train_df = self.encode_label(train_df, label_encoders)
        validation_df = self.encode_label(validation_df, label_encoders)
        test_df = self.encode_label(test_df, label_encoders)

        for split_name, df, kw in [
            ("training", train_df, dict()),
            ("validation", validation_df, dict()),
            (
                "test",
                test_df,
                dict(
                    save_identifier=True, save_time=True, save_only_future_outputs=False
                ),
            ),
        ]:
            logging.info(f"Creating TF time-series dataset for {split_name} split.")
            tf_ds = self.make_time_series_dataset(df, **kw)
            tf_ds.save(f"{save_path}/{self._name}/{split_name}")
            del tf_ds
            gc.collect()

        with open(f"{save_path}/{self._name}/target_scaler.pickle", "wb+") as file:
            pickle.dump(target_scalers, file, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    @abstractmethod
    def _name(self) -> str:
        raise NotImplementedError

    @property
    def _total_time_steps(self) -> int:
        return self.fixed_params.total_time_steps

    @cached_property
    def _id_column(self):
        return self._get_single_col_by_input_type(InputTypes.ID)

    @cached_property
    def _target_column(self):
        return self._get_single_col_by_input_type(InputTypes.TARGET)

    @cached_property
    def _real_inputs_columns(self):
        return self._get_cols_by_data_type(
            DataTypes.REAL_VALUED, {InputTypes.ID, InputTypes.TIME}
        )

    @cached_property
    def _categorical_inputs_columns(self):
        return self._get_cols_by_data_type(
            DataTypes.CATEGORICAL, {InputTypes.ID, InputTypes.TIME}
        )

    @cached_property
    def _time_column(self):
        return self._get_single_col_by_input_type(InputTypes.TIME)

    @cached_property
    def _inputs_static_columns(self):
        return self._get_cols_by_input_type(InputTypes.STATIC_INPUT)

    @cached_property
    def _inputs_observed_columns(self):
        return self._get_cols_by_input_type(InputTypes.OBSERVED_INPUT)

    @cached_property
    def _inputs_known_real_columns(self):
        return self._get_cols_by_input_type(
            InputTypes.KNOWN_INPUT, {DataTypes.CATEGORICAL, DataTypes.DATE}
        )

    @cached_property
    def _inputs_known_categorical_columns(self):
        return self._get_cols_by_input_type(
            InputTypes.KNOWN_INPUT, {DataTypes.REAL_VALUED, DataTypes.DATE}
        )

    @property
    def _num_encoder_steps(self) -> int:
        return self.fixed_params.num_encoder_steps

    @abstractmethod
    def _read_raw_csv(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, real_scalers: Any) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def _column_schema(self) -> Dict[str, SchemaEntry | List[SchemaEntry]]:
        raise NotImplementedError

    @abstractmethod
    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into training, validation and test dataframes."""
        raise NotImplementedError

    @abstractmethod
    def _fit_scalers(self, df: pd.DataFrame) -> Tuple[Any, Any]:
        """Scalers, are the ones, applied to real inputs and targets respectivelly."""
        raise NotImplementedError

    def _get_single_col_by_input_type(self, input_type: InputTypes) -> str:
        col = self._get_cols_by_input_type(input_type)
        if len(col) != 1:
            raise ValueError("Invalid number of columns for {}".format(input_type))
        return col[0]

    def _get_cols_by_input_type(
        self,
        input_type: InputTypes,
        excluded_data_types: Sequence[DataTypes] | None = None,
    ) -> List[str]:
        if excluded_data_types is None:
            excluded_data_types = []

        def filter_func(i: SchemaEntry | List[SchemaEntry]) -> bool:
            if isinstance(i, List):
                return any(map(filter_func, i))
            return i.input_type == input_type and i.data_type not in excluded_data_types

        return list(filter_dict(self._column_schema, value_filter=filter_func).keys())

    def _get_cols_by_data_type(
        self,
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

        return list(filter_dict(self._column_schema, value_filter=filter_func).keys())

    @abstractmethod
    def _fit_label_encoders(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
        raise NotImplementedError

    def make_time_series_dataset(
        self,
        df: pd.DataFrame,
        *,
        save_only_future_outputs: bool = True,
        save_identifier: bool = False,
        save_time: bool = False,
    ) -> tf.data.Dataset:
        col_mapping = {
            "identifier": [self._id_column],
            "time": [self._time_column],
            "outputs": [self._target_column],
            "inputs_static": self._inputs_static_columns,
            "inputs_known_categorical": self._inputs_known_categorical_columns,
            "inputs_known_real": self._inputs_known_real_columns,
            "inputs_observed": self._inputs_observed_columns,
        }

        logging.info(f"Creating TF dataset to be saved.")
        tf_ds = None
        df.sort_values(by=[self._id_column, self._time_column], inplace=True)
        for name, sliced in keras_pbar(df.groupby(self._id_column)):
            data_map: OrderedDict = OrderedDict()
            for k in col_mapping:
                cols = col_mapping[k]
                if (
                    (len(cols) == 0)
                    or (k == "identifier" and not save_identifier)
                    or (k == "time" and not save_time)
                ):
                    continue
                arr = sliced[cols].to_numpy()
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)

                data_map[k] = arr

            for k, v in data_map.items():
                v = np.stack(
                    list(
                        tf.keras.utils.timeseries_dataset_from_array(
                            v,
                            targets=None,
                            sequence_length=self._total_time_steps,
                            batch_size=None,
                        ).as_numpy_iterator()
                    )
                )
                if k in ("identifier", "inputs_static"):
                    # No need to save time steps for ids or static inputs, they do not vary.
                    v = v[:, 0]
                # This will cause issue for batching.
                if k == "outputs" and save_only_future_outputs:
                    v = v[:, self._num_encoder_steps :]
                data_map[k] = v

            data_map = dict(**data_map)
            tf_ds_i = tf.data.Dataset.from_tensor_slices(data_map)
            tf_ds = concatenate_datasets(tf_ds, tf_ds_i)

        return tf_ds

    def encode_label(
        self, df: pd.DataFrame, label_encoder: Dict[str, LabelEncoder]
    ) -> pd.DataFrame:
        output = df.copy()

        for col in self._categorical_inputs_columns:
            string_df = df[col].apply(str)
            output[col] = label_encoder[col].transform(string_df)
        return output


class ElectricityExperiment(Experiment):
    @property
    def _name(self) -> str:
        return "electricity"

    @property
    def fixed_params(self) -> FixedParams:
        return FixedParams(
            num_encoder_steps=7 * 24,
            total_time_steps=8 * 24,
            num_outputs=1,
            known_categories_sizes=[],
            static_categories_sizes=[369],
        )

    @property
    def _column_schema(self) -> Dict[str, SchemaEntry | List[SchemaEntry]]:
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

    def _read_raw_csv(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading electricity dataset from {path}")
        # This code was copy-pasted from original implementation, and I have very little idea
        # what is it doing.
        df = read_csv(path, index_col=0, sep=";", decimal=",")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Used to determine the start and end dates of a series
        df = df.resample("1h").mean().replace(0.0, np.nan)

        earliest_time = df.index.min()

        df_list = []

        for label in keras_pbar(df, n=369):
            column = df[label]

            start_date = min(column.fillna(method="ffill").dropna().index)
            end_date = max(column.fillna(method="bfill").dropna().index)

            active_range = (column.index >= start_date) & (column.index <= end_date)
            column = column[active_range].fillna(0.0)

            tmp = pd.DataFrame({"power_usage": column})
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

        return df

    def _split_data(
        self,
        df: pd.DataFrame,
        validation_boundary: int = 1315,
        test_boundary: int = 1339,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = df["days_from_start"]
        train_df = df.loc[index < validation_boundary]
        validation_df = df.loc[
            (index >= validation_boundary - 7) & (index < test_boundary)
        ]
        test_df = df.loc[index >= test_boundary - 7]

        return train_df, validation_df, test_df

    def _fit_scalers(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, StandardScaler], Dict[str, StandardScaler]]:
        real_scalers = dict()
        target_scalers = dict()
        logging.debug("Fitting scalers.")
        for identifier, sliced in keras_pbar(df.groupby(self._id_column)):
            if len(sliced) >= self._total_time_steps:
                data = sliced[self._real_inputs_columns].values
                targets = sliced[[self._target_column]].values
                real_scalers[identifier] = StandardScaler().fit(data)
                target_scalers[identifier] = StandardScaler().fit(targets)

        return real_scalers, target_scalers

    def _normalize_data(
        self,
        df: pd.DataFrame,
        real_scalers: Dict[str, StandardScaler],
    ) -> pd.DataFrame:
        df_list = []
        for identifier, sliced in keras_pbar(df.groupby(self._id_column)):
            if len(sliced) >= self._total_time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[self._real_inputs_columns] = real_scalers[
                    identifier
                ].transform(sliced_copy[self._real_inputs_columns].values)
                df_list.append(sliced_copy)

        return pd.concat(df_list)

    def _fit_label_encoders(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
        from sklearn.preprocessing import LabelEncoder

        num_classes = dict()
        categorical_scalers = dict()
        for col in keras_pbar(self._categorical_inputs_columns):
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            num_classes[col] = srs.nunique()
            categorical_scalers[col] = LabelEncoder().fit(srs.values)

        logging.debug(f"{num_classes = }")
        return categorical_scalers

    def make_target_scaler(self, save_path) -> TargetScaler:
        with open(f"{save_path}/electricity/target_scaler.pickle", "rb") as file:
            target_scalers: Dict[str, StandardScaler] = pickle.load(
                file, fix_imports=True
            )

        def scale(entity_id: str, values: np.ndarray) -> np.ndarray:
            return target_scalers[entity_id].inverse_transform(values)

        return scale


class FavoritaExperiment(Experiment):
    @property
    def fixed_params(self) -> FixedParams:
        return FixedParams(
            total_time_steps=120,
            num_encoder_steps=90,
            known_categories_sizes=[],
            static_categories_sizes=[],
            num_outputs=1,
        )

    def make_target_scaler(self, save_path) -> TargetScaler:
        with open(f"{save_path}/favorita/target_scaler.pickle", "rb") as file:
            scaler: Tuple[float, float] = pickle.load(file, fix_imports=True)

        mean, std = scaler

        def scale(_, values: np.ndarray) -> np.ndarray:
            return (values - mean) / std

        return scale

    @property
    def _name(self) -> str:
        return "favorita"

    @property
    def _column_schema(self) -> Dict[str, SchemaEntry | List[SchemaEntry]]:
        return {
            "traj_id": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.ID),
            "date": SchemaEntry(DataTypes.DATE, InputTypes.TIME),
            "log_sales": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.TARGET),
            "onpromotion": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            "transactions": SchemaEntry(
                DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT
            ),
            # "oil": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            "day_of_week": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            "day_of_month": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            "month": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            "national_hol": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            "regional_hol": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            "local_hol": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            "open": SchemaEntry(DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            "item_nbr": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "store_nbr": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "city": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "state": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "type": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "cluster": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "family": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "class": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
            "perishable": SchemaEntry(DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        }

    def _read_raw_csv(self, path: str) -> pd.DataFrame:
        # load temporal data
        temporal = read_csv(f"{path}/train.csv", index_col=0)
        store_info = read_csv(f"{path}/stores.csv", index_col=0)
        # oil = read_csv(f"{path}/oil.csv", index_col=0).iloc[:, 0]
        holidays = read_csv(f"{path}/holidays_events.csv")
        items = read_csv(f"{path}/items.csv", index_col=0)
        transactions = read_csv(f"{path}/transactions.csv")

        # Take first 6 months of data
        temporal["date"] = pd.to_datetime(temporal["date"])

        # Extract only a subset of data to save/process for efficiency
        start_date = datetime.datetime(2015, 1, 1)
        end_date = datetime.datetime(2016, 6, 1)
        temporal = temporal[(temporal["date"] >= start_date)]
        temporal = temporal[(temporal["date"] <= end_date)]
        dates = temporal["date"].unique()

        # Add trajectory identifier
        temporal["traj_id"] = (
            temporal["store_nbr"].apply(str) + "_" + temporal["item_nbr"].apply(str)
        )
        temporal["unique_id"] = temporal["traj_id"] + "_" + temporal["date"].apply(str)

        # Remove all IDs with negative returns
        logging.debug("Removing returns data")
        min_returns = temporal["unit_sales"].groupby(temporal["traj_id"]).min()
        valid_ids = set(min_returns[min_returns >= 0].index)
        selector = temporal["traj_id"].apply(lambda i: i in valid_ids)
        new_temporal = temporal[selector].copy()
        del temporal
        gc.collect()
        temporal = new_temporal
        temporal["open"] = 1

        # Resampling
        logging.debug("Resampling to regular grid")
        resampled_dfs = []
        for traj_id, raw_sub_df in keras_pbar(temporal.groupby("traj_id")):
            sub_df = raw_sub_df.set_index("date", drop=True).copy()
            sub_df = sub_df.resample("1d").last()
            sub_df["date"] = sub_df.index
            sub_df[["store_nbr", "item_nbr", "onpromotion"]] = sub_df[
                ["store_nbr", "item_nbr", "onpromotion"]
            ].fillna(method="ffill")
            sub_df["open"] = sub_df["open"].fillna(
                0
            )  # flag where sales data is unknown
            sub_df["log_sales"] = np.log(sub_df["unit_sales"])
            resampled_dfs.append(sub_df.reset_index(drop=True))

        new_temporal = pd.concat(resampled_dfs, axis=0)
        del temporal
        gc.collect()
        temporal = new_temporal

        # logging.debug("Adding oil")
        # oil.name = "oil"
        # oil.index = pd.to_datetime(oil.index)
        # logging.debug("-" * 100)
        # logging.debug(f"{dates = }")
        # logging.info(f"{oil.index = }")
        # logging.debug("-" * 100)
        # temporal = temporal.join(
        #    oil.loc[dates].fillna(method="ffill"), on="date", how="left"
        # )
        # temporal["oil"] = temporal["oil"].fillna(-1)

        logging.debug("Adding store info")
        temporal = temporal.join(store_info, on="store_nbr", how="left")

        logging.debug("Adding item info")
        temporal = temporal.join(items, on="item_nbr", how="left")

        transactions["date"] = pd.to_datetime(transactions["date"])
        temporal = temporal.merge(
            transactions,
            left_on=["date", "store_nbr"],
            right_on=["date", "store_nbr"],
            how="left",
        )
        temporal["transactions"] = temporal["transactions"].fillna(-1)

        # Additional date info
        temporal["day_of_week"] = pd.to_datetime(temporal["date"].values).dayofweek
        temporal["day_of_month"] = pd.to_datetime(temporal["date"].values).day
        temporal["month"] = pd.to_datetime(temporal["date"].values).month

        # Add holiday info
        logging.debug("Adding holidays")
        holiday_subset = holidays[holidays["transferred"].apply(lambda x: not x)].copy()
        holiday_subset.columns = [
            s if s != "type" else "holiday_type" for s in holiday_subset.columns
        ]
        holiday_subset["date"] = pd.to_datetime(holiday_subset["date"])
        local_holidays = holiday_subset[holiday_subset["locale"] == "Local"]
        regional_holidays = holiday_subset[holiday_subset["locale"] == "Regional"]
        national_holidays = holiday_subset[holiday_subset["locale"] == "National"]

        temporal["national_hol"] = temporal.merge(
            national_holidays, left_on=["date"], right_on=["date"], how="left"
        )["description"].fillna("")
        temporal["regional_hol"] = temporal.merge(
            regional_holidays,
            left_on=["state", "date"],
            right_on=["locale_name", "date"],
            how="left",
        )["description"].fillna("")
        temporal["local_hol"] = temporal.merge(
            local_holidays,
            left_on=["city", "date"],
            right_on=["locale_name", "date"],
            how="left",
        )["description"].fillna("")

        temporal.sort_values("unique_id", inplace=True)
        return temporal

    def _fit_label_encoders(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
        categorical_scalers = {}
        num_classes = []
        id_set = set(list(df[self._id_column].unique()))
        valid_idx = df["traj_id"].apply(lambda x: x in id_set)
        for col in self._categorical_inputs_columns:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str).loc[valid_idx]
            categorical_scalers[col] = LabelEncoder().fit(srs.values)

            num_classes.append(srs.nunique())
        return categorical_scalers

    def _fit_scalers(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
        # Format real scalers
        real_scalers = {}
        for col in [
            # "oil",
            "transactions",
            "log_sales",
        ]:
            real_scalers[col] = (df[col].mean(), df[col].std())
        target_scaler = (df[self._target_column].mean(), df[self._target_column].std())
        return real_scalers, target_scaler

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        valid_boundary = datetime.datetime(2015, 12, 1)

        time_steps = self._total_time_steps
        lookback = self.fixed_params.num_encoder_steps
        forecast_horizon = time_steps - lookback

        df["date"] = pd.to_datetime(df["date"])
        df_lists = {"train": [], "valid": [], "test": []}
        for _, sliced in keras_pbar(df.groupby("traj_id")):
            index = sliced["date"]
            train = sliced.loc[index < valid_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            valid = sliced.iloc[train_len - lookback : valid_len, :]
            test = sliced.iloc[valid_len - lookback : valid_len + forecast_horizon, :]

            sliced_map = {"train": train, "valid": valid, "test": test}

            for k in sliced_map:
                item = sliced_map[k]

                if len(item) >= time_steps:
                    df_lists[k].append(item)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

        train = dfs["train"]
        # Filter out identifiers not present in training (i.e. cold-started items).
        identifiers = list(df[self._id_column].unique())

        def filter_ids(frame):
            ids = set(identifiers)
            index = frame["traj_id"]
            return frame.loc[index.apply(lambda x: x in ids)]

        valid = filter_ids(dfs["valid"])
        test = filter_ids(dfs["test"])

        return train, valid, test

    def _normalize_data(
        self,
        df: pd.DataFrame,
        real_scalers: Dict[str, Tuple[float, float]],
    ) -> pd.DataFrame:
        output = df.copy()

        for col in [
            "log_sales",
            # "oil",
            "transactions",
        ]:
            mean, std = real_scalers[col]
            output[col] = (df[col] - mean) / std

            if col == "log_sales":
                output[col] = output[col].fillna(0.0)  # mean imputation

        return output


def concatenate_datasets(
    ds1: tf.data.Dataset | None, ds2: tf.data.Dataset
) -> tf.data.Dataset:
    if ds1 is None:
        return ds2
    else:
        return ds1.concatenate(ds2)


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    logging.debug(f"Reading {path}")
    if not cudf_available:
        return pd.read_csv(path, **kwargs, engine="pyarrow")
    else:
        return pd.read_csv(path, **kwargs)


electricity_experiment = ElectricityExperiment()
favorita_experiment = FavoritaExperiment()
