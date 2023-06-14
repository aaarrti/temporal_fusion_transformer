import jax.numpy as jnp
import jax
import chex
from temporal_fusion_transformer.flax_.quantile_loss import (
    quantile_loss,
    quantile_rmse,
    # QuantileLoss,
    # QuantileRMSE,
)
from tests.constants import PRNG_SEED

quantiles = jnp.asarray([0.1, 0.5, 0.9])
prng_key = jax.random.PRNGKey(PRNG_SEED)


class QuantileLossTest(chex.TestCase):
    @chex.variants(
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
    )
    def test_loss_fn(self):
        y_true = jax.random.uniform(prng_key, (8, 24, 1))
        y_pred = jnp.random.uniform(prng_key, (8, 24, 3))
        loss = quantile_loss(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(loss)
        chex.assert_rank(loss, 1)
        chex.assert_shape(loss, (8,))

    def test_loss_converges_to_0(self):
        y_true = jnp.ones((8, 24, 1))
        y_pred = jnp.ones((8, 24, 3))

        loss = quantile_loss(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(loss)
        chex.assert_rank(loss, 1)
        chex.assert_shape(loss, (8,))
        chex.assert_tree_all_close(loss, 0.0)

    # def test_loss_fn_wrapper(self):
    #    # TODO: test CLU container


class QuantileRMSETest(chex.TestCase):
    def test_rmse(self):
        y_true = jax.random.uniform(prng_key, (8, 24, 1))
        y_pred = jax.random.uniform(prng_key, (8, 24, 3))
        rmse = quantile_rmse(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(rmse)
        chex.assert_rank(rmse, 1)
        chex.assert_shape(rmse, (8,))

    def test_rmse_converges_to_0(self):
        y_true = jnp.ones((8, 24, 1))
        y_pred = jnp.ones((8, 24, 3))

        rmse = quantile_rmse(y_true, y_pred, quantiles)
        chex.assert_tree_all_finite(rmse)
        chex.assert_rank(rmse, 1)
        chex.assert_shape(rmse, (8,))
        chex.assert_tree_all_close(rmse, 0.0)

    # def test_metric_wrapper(self):
    #    # TODO: test CLU container
