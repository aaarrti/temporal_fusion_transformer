import jax.random
import jax.numpy as jnp

from temporal_fusion_transformer.train_lib.loss_fn import quantile_pinball_loss, pinball_loss

key = jax.random.PRNGKey(42)


def test_loss_fn():
    y_true = jnp.ones((8, 2, 1), dtype=jnp.float32)
    y_pred = jnp.ones((8, 2, 1), dtype=jnp.float32)
    loss = pinball_loss(y_true, y_pred, 0.5)
    assert loss.shape == ()


def test_quantile_loss_fn():
    y_true = jnp.ones((8, 2, 1), dtype=jnp.float32)
    y_pred = jnp.ones((8, 2, 1, 3), dtype=jnp.float32)
    loss = quantile_pinball_loss(y_true, y_pred)
    assert loss.shape == ()
