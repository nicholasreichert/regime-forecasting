from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval.metrics import (
    mae,
    rmse,
    directional_accuracy,
    spearman_corr,
    top_decile_hit_rate, 
)
from src.eval.subsets import high_vol_mask, top_quantile_mask, apply_mask
from src.eval.walk_forward import walk_forward_splits


def evaluate_model(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model,
    train_years: int,
    test_years: int,
    step_years: int,
) -> Dict[str, float]:
    """
    walk-forward eval

    return targets: rmse, mae, directional_accuracy.
    volatility targets (absret/sqret):  rmse, mae, spearman, top_decile_hit_rate,
      plus stress subset metrics:
        - *_hv_now: top decile of ret_vol_20 within each test window (current stress)
        - *_hv_fut: top decile of y_true within each test window (future spike stress)
    """
    is_vol_target = ("_absret_" in target) or ("_sqret_" in target)

    if is_vol_target:
        metrics: Dict[str, List[float]] = {
            "rmse": [],
            "mae": [],
            "spearman": [],
            "top_decile_hit": [],
            "rmse_hv_now": [],
            "mae_hv_now": [],
            "rmse_hv_fut": [],
            "mae_hv_fut": [],
        }
    else:
        metrics = {
            "rmse": [],
            "mae": [],
            "directional_accuracy": [],
        }

    for split in walk_forward_splits(df.index, train_years, test_years, step_years):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        # Fit / predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Full-sample metrics
        metrics["rmse"].append(rmse(y_test, y_pred))
        metrics["mae"].append(mae(y_test, y_pred))

        if is_vol_target:
            metrics["spearman"].append(spearman_corr(y_test, y_pred))
            metrics["top_decile_hit"].append(top_decile_hit_rate(y_test, y_pred))

            yt = np.asarray(y_test, dtype=float)
            yp = np.asarray(y_pred, dtype=float)

            # hv_now: current stress defined by ret_vol_20 within THIS test window
            hv_now = high_vol_mask(test, vol_col="ret_vol_20", q=0.9)
            yt_now, yp_now = apply_mask(yt, yp, hv_now)
            if len(yt_now) > 0:
                metrics["rmse_hv_now"].append(rmse(yt_now, yp_now))
                metrics["mae_hv_now"].append(mae(yt_now, yp_now))

            # hv_fut: future stress defined by top decile of y_true within THIS test window
            hv_fut = top_quantile_mask(yt, q=0.9)
            yt_fut, yp_fut = apply_mask(yt, yp, hv_fut)
            if len(yt_fut) > 0:
                metrics["rmse_hv_fut"].append(rmse(yt_fut, yp_fut))
                metrics["mae_hv_fut"].append(mae(yt_fut, yp_fut))
        else:
            metrics["directional_accuracy"].append(directional_accuracy(y_test, y_pred))

    # safe aggregation: if a metric list is empty, return nan
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(np.mean(v)) if len(v) else float("nan")
    return out
