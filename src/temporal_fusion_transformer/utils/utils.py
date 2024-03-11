from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

if TYPE_CHECKING:
    import holoviews as hv

    Array = np.ndarray | jax.Array


def time_series_to_array(ts: Array) -> np.ndarray:
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
    x: Array,
    total_time_steps: int,
    arr_factory: Callable[[Array], np.ndarray] = partial(np.asarray, dtype=jnp.float32),
) -> np.ndarray:
    """
    Converts raw dataframe from a 2-D tabular format to a batched 3-D array to feed into model.

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


def unpack_xy(arr: Array, encoder_steps: int, n_targets: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x_id = arr.shape[-1] - n_targets
    x = arr[..., :x_id]
    y = arr[:, encoder_steps:, x_id:]
    return x, y


def plot_predictions_vs_real(dataframe: pl.DataFrame, **kwargs) -> hv.Layout:
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
    from bokeh.models import DatetimeTickFormatter

    xformatter = DatetimeTickFormatter(months="%b %Y")
    kwargs = dict(x="ts", autorange="x", grid=True, legend=True, xformatter=xformatter, **kwargs)
    area = dataframe.plot.area(y="yhat_up", y2="yhat_low", fill_alpha=0.4, **kwargs)
    p1 = dataframe.plot.line(y="y", color="gray", **kwargs)
    s1 = dataframe.plot.scatter(y="y", color="gray", **kwargs)
    p2 = dataframe.plot.line(y="yhat", color="blue", **kwargs).opts(color="blue")
    s2 = dataframe.plot.scatter(y="yhat", color="blue", **kwargs).opts(color="blue")
    return (area * (p1 * s1)) * (p2 * s2)


def plot_feature_importance(
    dataframe: pl.DataFrame, title: str, fill_alpha: float | None = 0.8, **kwargs
) -> hv.Layout:
    """
    You probably would want to do min-max normalization of features beforehand.

    Parameters
    ----------
    dataframe: must contain `ts` column and features to plot.
    title
    fill_alpha

    Returns
    -------
    """
    from bokeh.models import DatetimeTickFormatter

    xformatter = DatetimeTickFormatter(months="%b %Y")
    features = dataframe.drop("ts").columns

    return dataframe.plot.area(
        x="ts",
        y=0,
        y2=features,
        xformatter=xformatter,
        legend=True,
        grid=True,
        fill_alpha=fill_alpha,
        title=title,
        stacked=False,
        **kwargs,
    )
