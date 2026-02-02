from __future__ import annotations

import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true-y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

