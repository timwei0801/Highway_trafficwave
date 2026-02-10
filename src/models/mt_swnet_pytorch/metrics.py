# -- coding: utf-8 --
import numpy as np


def metric(pred, label):
    """Compute MAE, RMSE, MAPE with zero-masking.

    Args:
        pred:  numpy array
        label: numpy array
    Returns:
        mae, rmse, mape
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape
