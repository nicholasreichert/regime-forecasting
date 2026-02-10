from __future__ import annotations

import numpy as np
import pandas as pd


def _nan_safe_top_quantile_threshold(values: np.ndarray, q: float) -> float | None:
    """Return the nan-safe top-q quantile threshold or None if all values are NaN."""
    if np.all(np.isnan(values)):
        return None
    return float(np.nanquantile(values, q))


def high_vol_mask(
    test_df: pd.DataFrame,
    vol_col: str = "ret_vol_20",
    q: float = 0.9,
) -> np.ndarray:
    """Boolean mask for rows in ``test_df`` in the top-q quantile of ``vol_col``.

    If the column is entirely NaN, a all-False mask is returned.
    """
    if vol_col not in test_df.columns:
        raise KeyError(f"vol_col='{vol_col}' not found in test_df columns")

    # Avoid an unnecessary copy when possible
    vol = test_df[vol_col].to_numpy(dtype="float64", copy=False)
    thr = _nan_safe_top_quantile_threshold(vol, q)
    if thr is None:
        return np.zeros(test_df.shape[0], dtype=bool)
    return vol >= thr


def top_quantile_mask(values: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Boolean mask for entries in a 1D array that are in the top-q quantile.

    If all entries are NaN, an all-False mask is returned.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1:
        raise ValueError("top_quantile_mask expects a 1D array.")

    thr = _nan_safe_top_quantile_threshold(values, q)
    if thr is None:
        return np.zeros(values.shape[0], dtype=bool)
    return values >= thr


def apply_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a boolean mask to ``y_true`` and ``y_pred``.

    Ensures consistent lengths and returns masked views (no copies when possible).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if mask.shape[0] != y_true.shape[0]:
        raise ValueError("Mask must have same length as y_true/y_pred.")

    return y_true[mask], y_pred[mask]