from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable
from collections.abc import Iterator
from functools import partial
import holoviews
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from bokeh.models import DatetimeTickFormatter
from flax import struct
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


@struct.dataclass
class FeatureImportance:
    historical_flags: jax.Array
    future_flags: jax.Array


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


def timeseries_from_array(
    x: np.ndarray | jnp.ndarray,
    total_time_steps: int,
    arr_factory: Callable[[np.ndarray], np.ndarray] = partial(np.asarray, dtype=jnp.float32),
) -> np.ndarray:
    """
    Converts raw dataframe from a 2-D tabular format to a batched 3-D array to feed into Keras model.

    Parameters
    -------

    x:
        2D array.
    total_time_steps:

    arr_factory:



    Returns
    -------

    arr:
        Batched Numpy array with shape=(?, self.time_steps, self.input_size)

    """
    x = arr_factory(x)
    time_steps = len(x)
    if time_steps < total_time_steps:
        raise ValueError("time_steps < total_time_steps")

    return np.stack(
        [x[i : time_steps - (total_time_steps - 1) + i, :] for i in range(total_time_steps)], axis=1
    )


def unpack_xy(
    arr: np.ndarray, encoder_steps: int, n_targets: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    x_id = arr.shape[-1] - n_targets
    x = arr[..., :x_id]
    y = arr[:, encoder_steps:, x_id:]
    return x, y


def split_dataframe(
    dataframe: pl.DataFrame, test_boundary: datetime | date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        dataframe.filter(pl.col("ts") < test_boundary),
        dataframe.filter(pl.col("ts") >= test_boundary),
    )


def plot_split(
    dataframe: pl.DataFrame,
    validation_boundary: date | datetime,
    **kwargs,
) -> holoviews.Layout:
    """
    Parameters
    ----------
    dataframe:
        Must have columns `y` and `ts`
    validation_boundary
    kwargs

    Returns
    -------

    """
    xformatter = DatetimeTickFormatter(months="%b %Y")
    train_dataframe, validation_dataframe = split_dataframe(dataframe, validation_boundary)
    train_dataframe = train_dataframe.with_columns(split=pl.lit("training"))
    validation_dataframe = validation_dataframe.with_columns(split=pl.lit("validation"))
    dataframe = pl.concat([train_dataframe, validation_dataframe])
    kw = dict(y="y", x="ts", xformatter=xformatter, by="split", legend=True, grid=True, **kwargs)
    return dataframe.plot.line(**kw) * dataframe.plot.scatter(**kw)


def plot_predictions_vs_real(dataframe: pl.DataFrame, **kwargs) -> holoviews.Layout:
    """

    Parameters
    ----------
    dataframe: Dataframe with columns:
     - ts
     - y
     - yhat
     - yhat_low
     - yhat_up

    Returns
    -------
    """

    xformatter = DatetimeTickFormatter(months="%b %Y")

    kwargs = dict(x="ts", autorange="x", grid=True, legend=True, xformatter=xformatter, **kwargs)

    area = dataframe.plot.area(y="yhat_up", y2="yhat_low", **kwargs)

    p1 = dataframe.plot.line(y="y", color="gray", **kwargs)
    s1 = dataframe.plot.scatter(y="y", color="gray", **kwargs)

    p2 = dataframe.plot.line(y="yhat", color="blue", **kwargs).opts(color="blue")
    s2 = dataframe.plot.scatter(y="yhat", color="blue", **kwargs).opts(color="blue")

    return (area * (p1 * s1)) * (p2 * s2)


def plot_feature_importance(
    ts: list[date],
    explanations: FeatureImportance,
    feature_names: list[str],
) -> holoviews.Layout:
    """
    Parameters
    ----------
    ts
    explanations:
        Must contain flattened 2D sequences, not time series!!!
    feature_names

    Returns
    -------

    """
    xformatter = DatetimeTickFormatter(months="%b %Y")

    data = {
        f"{name}_importance": np.concatenate(
            [explanations.historical_flags[..., i], explanations.future_flags[..., i]]
        )
        for i, name in enumerate(feature_names)
    }

    explanations_df = pl.DataFrame(
        {
            "ts": ts,
            **data,
        }
    )

    return explanations_df.plot.area(
        x="ts",
        y=list(data.keys()),
        xformatter=xformatter,
        legend=True,
        grid=True,
    )
