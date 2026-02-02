from __future__ import annotations

import numpy as np
import pandas as pd


def high_vol_mask(
    test_df: pd.DataFrame,
    vol_col: str = "ret_vol_20",
    q: float = 0.9,
) -> np.ndarray:
    # boolean mask for rows in test_df in top-q quantile of vol_col
    if vol_col not in test_df.columns:
        raise KeyError(f"vol_col='{vol_col}' not found in test_df columns")

    vol = test_df[vol_col].to_numpy(dtype=float)
    if np.all(np.isnan(vol)):
        return np.zeros(len(test_df), dtype=bool)

    thr = np.nanquantile(vol, q)
    return vol >= thr


def top_quantile_mask(values: np.ndarray, q: float = 0.9) -> np.ndarray:
    # boolean mask for values in top-q quantile of the array
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("top_quantile_mask expects a 1D array.")

    if np.all(np.isnan(values)):
        return np.zeros(len(values), dtype=bool)

    thr = np.nanquantile(values, q)
    return values >= thr


def apply_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != len(y_true):
        raise ValueError("Mask must have same length as y_true/y_pred.")
    return y_true[mask], y_pred[mask]