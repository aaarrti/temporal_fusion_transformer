import jax.numpy as jnp
import numpy as np
import jax
import chex
from temporal_fusion_transformer.src.training_flax import (
    quantile_loss,
    quantile_rmse,
)
from tests.constants import PRNG_SEED

quantiles = jnp.asarray([0.1, 0.5, 0.9])
prng_key = jax.random.PRNGKey(PRNG_SEED)


class QuantileLossTest(chex.TestCase):
    @chex.variants(
        with_jit=True,
        without_jit=True,
    )
    def test_loss_fn(self):
        y_true = jax.random.uniform(prng_key, (8, 24, 1))
        y_pred = jax.random.uniform(prng_key, (8, 24, 3))
        loss = self.variant(quantile_loss)(y_true, y_pred, np.asarray(quantiles))
        chex.assert_tree_all_finite(loss)
        chex.assert_rank(loss, 1)
        chex.assert_shape(loss, (8,))

    @chex.variants(
        with_jit=True,
        without_jit=True,
    )
    def test_loss_converges_to_0(self):
        y_true = jnp.ones((8, 24, 1))
        y_pred = jnp.ones((8, 24, 3))

        loss = self.variant(quantile_loss)(y_true, y_pred, np.asarray(quantiles))
        chex.assert_tree_all_finite(loss)
        chex.assert_rank(loss, 1)
        chex.assert_shape(loss, (8,))
        chex.assert_tree_all_close(loss, 0.0)


class QuantileRMSETest(chex.TestCase):
    @chex.variants(
        with_jit=True,
        without_jit=True,
    )
    def test_rmse(self):
        y_true = jax.random.uniform(prng_key, (8, 24, 1))
        y_pred = jax.random.uniform(prng_key, (8, 24, 3))
        rmse = self.variant(quantile_rmse)(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(rmse)
        chex.assert_rank(rmse, 1)
        chex.assert_shape(rmse, (8,))

    @chex.variants(
        with_jit=True,
        without_jit=True,
    )
    def test_rmse_converges_to_0(self):
        y_true = jnp.ones((8, 24, 1))
        y_pred = jnp.ones((8, 24, 3))

        rmse = self.variant(quantile_rmse)(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(rmse)
        chex.assert_rank(rmse, 1)
        chex.assert_shape(rmse, (8,))
        chex.assert_tree_all_close(rmse, 0.0)
