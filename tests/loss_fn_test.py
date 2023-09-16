import chex
import jax.numpy as jnp
import jax.random
from src.modeling.loss_fn import make_quantile_loss_fn, pinball_loss, quantile_loss

PRNG_KEY = jax.random.PRNGKey(0)


def test_loss_fn():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    loss = pinball_loss(y_true, y_pred, tau=0.5)
    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_tree_all_finite(loss)


def test_loss_converges_to_0():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = y_true + 0.0001

    loss = pinball_loss(y_true, y_pred, tau=0.5)
    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_trees_all_close(loss, 0, atol=0.01)


def test_quantile_loss():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (8, 24, 1, 3))
    loss = quantile_loss(y_true, y_pred, jnp.asarray([0.1, 0.5, 0.9]))

    chex.assert_rank(loss, 2)
    chex.assert_shape(loss, (8, 3))
    chex.assert_tree_all_finite(loss)


def test_make_quantile_loss_fn():
    loss_fn = make_quantile_loss_fn([0.1, 0.5, 0.9])
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (8, 24, 1, 3))
    loss = loss_fn(y_true, y_pred)

    chex.assert_rank(loss, 2)
    chex.assert_shape(loss, (8, 3))
    chex.assert_tree_all_finite(loss)
