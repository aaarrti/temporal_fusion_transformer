import chex
import jax.numpy as jnp
import jax.random

from temporal_fusion_transformer.src.loss_fn import quantile_pinball_loss

PRNG_KEY = jax.random.PRNGKey(0)
quantiles = jnp.asarray([0.1, 0.5, 0.9], jnp.float32)


def test_quantile_loss():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jax.random.uniform(PRNG_KEY, (8, 24, 1, 3))
    loss = quantile_pinball_loss(y_true, y_pred, quantiles)

    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_tree_all_finite(loss)


def test_loss_converges_to_0():
    y_true = jax.random.uniform(PRNG_KEY, (8, 24, 1))
    y_pred = jnp.stack([y_true, y_true, y_true], axis=-1)

    loss = quantile_pinball_loss(y_true, y_pred, quantiles)
    chex.assert_rank(loss, 1)
    chex.assert_shape(loss, (8,))
    chex.assert_trees_all_close(loss, 0.001, atol=0.01)
