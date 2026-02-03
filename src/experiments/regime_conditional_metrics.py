from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.eval.metrics import rmse, mae, spearman_corr, top_decile_hit_rate
from src.eval.subsets import top_quantile_mask, apply_mask

def compute_regime_conditional_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        oos_regimes: pd.DataFrame,
        is_vol_target: bool,
) -> pd.DataFrame:
    # compute metrics conditional on inferred regime at prediciton time
    
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "regime": oos_regimes["hard_state"],
        }
    ).dropna()

    out_rows = []

    hv_fut_mask = top_quantile_mask(df["y_true"].values, q=0.9)

    for k in sorted(df["regime"].unique()):
        mask = df["regime"] == k
        yt = df.loc[mask, "y_true"].values
        yp = df.loc[mask, "y_pred"].values

        row: Dict[str, float] = {
            "regime": int(k),
            "count": int(mask.sum()),
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
        }

        if is_vol_target:
            row["spearman"] = spearman_corr(yt, yp)
            row["top_decile_hit"] = top_decile_hit_rate(yt, yp)

            hv_fut_regime = hv_fut_mask & mask.values
            yt_fut, yp_fut = apply_mask(
                df["y_true"].values,
                df["y_pred"].values,
                hv_fut_regime,
            )

            row["rmse_hv_fut"] = rmse(yt_fut, yp_fut) if len(yt_fut) > 0 else np.nan
            row["hv_fut_frac"] = float(hv_fut_regime.mean())

        out_rows.append(row)

    return pd.DataFrame(out_rows)