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


def qlike(y_true, y_pred, floor_q: float = 0.01) -> float:
    """Mean QLIKE loss between volatility forecasts (Patton, 2011).

    Both inputs are volatilities; the loss is evaluated on the implied
    variances:

        QLIKE = v_true / v_pred - log(v_true / v_pred) - 1

    QLIKE is one of the two loss families (with MSE) that stay robust when the
    volatility target is a noisy proxy for latent volatility, so a ranking under
    QLIKE cannot be an artefact of proxy noise. Unlike MSE it penalises
    under-prediction far more heavily than over-prediction, which is the
    relevant asymmetry for risk applications.

    QLIKE is unbounded as the forecast approaches zero, so a forecast of exactly
    zero has infinite loss. That is the economically correct verdict but it makes
    the statistic useless for ranking, and at h=1 the proxy |r_{t+1}| is itself
    zero often enough to matter.

    Both the realization and the forecast are therefore floored at the
    ``floor_q`` quantile of the realized target. Flooring *both* sides keeps the
    loss exactly zero for a perfect forecast, which a one-sided floor does not;
    it amounts to evaluating on a scale where volatilities below the 1st
    percentile are treated as indistinguishable. The floor is applied for
    reporting only and never touches model fitting or selection.
    """
    losses, _ = qlike_losses(y_true, y_pred, floor_q=floor_q)
    if losses.size == 0:
        return float("nan")
    return float(np.mean(losses))


def qlike_losses(y_true, y_pred, floor_q: float = 0.01):
    """Per-observation QLIKE losses and the mask of observations kept.

    Shared with the Diebold-Mariano tests so that a significance test on QLIKE
    is testing the same quantity the results table reports. Keeping two
    independently-floored implementations made the h=1 tests degenerate: with a
    near-zero floor the loss differential is dominated by a handful of
    observations where the realization is essentially zero, and every comparison
    returns the same p-value regardless of the models involved.
    """
    y_true_arr, y_pred_arr = _validate_y_true_y_pred(y_true, y_pred)

    ok = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & (y_true_arr > 0)
    if ok.sum() == 0:
        return np.empty(0, dtype=float), ok

    floor = float(np.quantile(y_true_arr[ok], floor_q))
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-8

    v_true = np.maximum(y_true_arr[ok], floor) ** 2
    v_pred = np.maximum(y_pred_arr[ok], floor) ** 2

    ratio = v_true / v_pred
    return ratio - np.log(ratio) - 1.0, ok


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


