import jax.random
import jax.numpy as jnp
import chex
from temporal_fusion_transformer.src.quantile_loss import quantile_loss, make_quantile_loss_fn

quantiles = (0.1, 0.5, 0.9)
PRNG_KEY = jax.random.PRNGKey(0)


def test_loss_fn():
    y_true = jax.random.uniform(PRNG_KEY, (24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (24, 3))
    loss = quantile_loss(y_true, y_pred, quantiles=quantiles)
    chex.assert_rank(loss, 0)
    chex.assert_shape(loss, ())
    chex.assert_tree_all_finite(loss)


def test_loss_converges_to_0():
    y_true = jax.random.uniform(PRNG_KEY, (24, 1))
    y_pred = y_true + 0.0001

    loss = quantile_loss(y_true, y_pred, quantiles=quantiles)
    chex.assert_rank(loss, 0)
    chex.assert_shape(loss, ())
    chex.assert_trees_all_close(loss, 0, atol=0.01)


def test_loss_batched():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (8, 24, 3))
    loss = make_quantile_loss_fn(quantiles)(y_true, y_pred)

    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_tree_all_finite(loss)
