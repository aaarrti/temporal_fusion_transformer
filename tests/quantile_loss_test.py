import numpy as np
from temporal_fusion_transformer.src.quantile_loss import quantile_loss

quantiles = (0.1, 0.5, 0.9)


def test_loss_fn():
    y_true = np.random.default_rng(69).uniform(size=(24, 1))
    y_pred = np.random.default_rng(69).uniform(size=(24, 3))
    loss = quantile_loss(y_true, y_pred, quantiles=quantiles)
    assert not np.isnan(loss).any()
    assert np.ndim(loss) == 0
    assert loss.shape == ()


def test_loss_converges_to_0():
    y_true = np.ones((24, 1))
    y_pred = np.ones((24, 3))

    loss = quantile_loss(y_true, y_pred, quantiles=quantiles)
    assert np.ndim(loss) == 0
    assert loss.shape == ()
    np.testing.assert_allclose(loss, 0)
