from __future__ import annotations

import math
from typing import cast

import numpy as np
from scipy.stats import spearmanr


def _to_1d_float_array(x: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    """Convert input to a 1D float64 numpy array.

    Raises a ValueError if the array is empty after raveling.
    """
    arr = np.asarray(x, dtype="float64").ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _validate_y_true_y_pred(
    y_true: np.ndarray | list[float] | tuple[float, ...],
    y_pred: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and standardize ``y_true`` and ``y_pred``.

    Ensures both are 1D float64 arrays of the same length.
    """
    y_true_arr = _to_1d_float_array(y_true, "y_true")
    y_pred_arr = _to_1d_float_array(y_pred, "y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return y_true_arr, y_pred_arr


def spearman_corr(y_true, y_pred) -> float:
    """Spearman rank correlation between ``y_true`` and ``y_pred``.

    Returns 0.0 when one of the series is (almost) constant or
    when the correlation cannot be computed.
    """
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)

    if np.std(y_true_arr) < 1e-12 or np.std(y_pred_arr) < 1e-12:
        return 0.0

    # SciPy's type hints can be imprecise here; explicitly cast to float
    corr = cast(float, spearmanr(y_true_arr, y_pred_arr)[0])
    corr_f = corr
    return 0.0 if math.isnan(corr_f) else corr_f


def top_decile_hit_rate(y_true, y_pred, q: float = 0.9) -> float:
    """Fraction of true top-q observations captured by predicted top-q.

    If no predictions fall into the top-q quantile, returns 0.0.
    """
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)

    thr_true = np.quantile(y_true_arr, q)
    thr_pred = np.quantile(y_pred_arr, q)

    pred_top = y_pred_arr >= thr_pred
    if pred_top.sum() == 0:
        return 0.0

    true_top = y_true_arr >= thr_true
    return float(np.mean(true_top[pred_top]))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between ``y_true`` and ``y_pred``."""
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error between ``y_true`` and ``y_pred``."""
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def directional_accuracy(y_true, y_pred) -> float:
    """Share of observations where ``y_true`` and ``y_pred`` have same sign.

    Observations with ``y_true == 0`` are ignored. If no such observations
    exist, returns NaN.
    """
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)

    mask = y_true_arr != 0
    if mask.sum() == 0:
        return float("nan")

    return float(np.mean((y_true_arr[mask] > 0) == (y_pred_arr[mask] > 0)))


