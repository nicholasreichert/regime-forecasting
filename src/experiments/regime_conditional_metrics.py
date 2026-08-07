from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.eval.metrics import mae, rmse, spearman_corr, top_decile_hit_rate
from src.eval.subsets import apply_mask, top_quantile_mask


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

            # Three genuinely different quantities, previously collapsed into a
            # single ambiguously-named "hv_fut_frac" that reported the third one
            # (so the columns summed to 0.1 rather than to 1, and reading it as a
            # within-regime rate overstated concentration by an order of
            # magnitude).
            n_regime = int(mask.sum())
            n_hv_in_regime = int(hv_fut_regime.sum())
            n_hv_total = int(hv_fut_mask.sum())

            # P(stress | regime): how stressed this regime actually is
            row["hv_fut_rate_within_regime"] = (
                n_hv_in_regime / n_regime if n_regime else np.nan
            )
            # P(regime | stress): how concentrated stress is in this regime
            row["hv_fut_share_of_all_stress"] = (
                n_hv_in_regime / n_hv_total if n_hv_total else np.nan
            )
            # lift over the unconditional 10% base rate; 1.0 means no information
            base_rate = n_hv_total / len(df) if len(df) else np.nan
            row["hv_fut_lift"] = (
                (n_hv_in_regime / n_regime) / base_rate
                if n_regime and base_rate
                else np.nan
            )

        out_rows.append(row)

    return pd.DataFrame(out_rows)