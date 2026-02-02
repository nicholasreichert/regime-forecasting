from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

def spearman_corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    
    corr, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)

def top_decile_hit_rate(y_true, y_pred, q: float = 0.9) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    thr_true = np.quantile(y_true, q)
    thr_pred = np.quantile(y_pred, q)

    pred_top = y_pred >= thr_pred
    if pred_top.sum() == 0:
        return 0.0
    
    true_top = y_true >= thr_true
    return float(np.mean(true_top[pred_top]))

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

    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    
    return float(np.mean((y_true[mask] > 0) == (y_pred[mask] > 0)))


