from temporal_fusion_transformer.src.quantile_metrics import QuantileRMSE
import chex
from keras_core import ops
import numpy as np


def test_quantile_rmse():
    y_true = ops.ones(shape=(8, 24, 1))
    y_pred = ops.ones(shape=(8, 24, 1, 3))

    rmse = QuantileRMSE(tau=0.9, quantile_index=0)

    rmse.update_state(y_true, y_pred)

    result = rmse.result()

    np.testing.assert_allclose(result, 0.0)


def test_quantile_rmse_multi_output():
    y_true = ops.ones(shape=(8, 24, 2))
    y_pred = ops.ones(shape=(8, 24, 2, 3))

    rmse = QuantileRMSE(tau=0.9, quantile_index=1, output_index=1)

    rmse.update_state(y_true, y_pred)

    result = rmse.result()

    np.testing.assert_allclose(result, 0.0)
