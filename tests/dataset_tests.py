import jax.random
import numpy as np

from temporal_fusion_transformer.src.datasets.utils import timeseries_from_array, unpack_xy
import chex

key = jax.random.PRNGKey(42)


def test_timeseries_from_array():
    x = jax.random.uniform(key, shape=(128, 5))
    ts = timeseries_from_array(x, lags=12)

    assert ts.dtype == np.float32
    chex.assert_rank(ts, 3)
    chex.assert_shape(ts, (117, 12, 5))


def test_unpack_xy():
    x = jax.random.uniform(key, shape=(117, 12, 6))
    x, y = unpack_xy(x, encoder_steps=9)

    assert x.dtype == np.float32
    assert y.dtype == np.float32

    chex.assert_rank(x, 3)
    chex.assert_rank(y, 3)
    chex.assert_shape(x, (117, 12, 5))
    chex.assert_shape(y, (117, 3, 1))
