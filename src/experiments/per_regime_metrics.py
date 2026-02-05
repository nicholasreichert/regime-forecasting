from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.eval.metrics import mae, rmse, spearman_corr, top_decile_hit_rate


@dataclass(frozen=True)
class PerRegimeMetricsConfig:
    oos_probs_csv: Path          # .../oos_regime_probs.csv
    oos_predictions_csv: Path    # .../oos_predictions.csv
    out_csv: Path                # .../per_regime_metrics.csv
    # define "future stress" as top q of y_true over the full OOS period
    hv_fut_q: float = 0.9


def _load_probs(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        # assume first column is date/index, but don't overwrite its dtype
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col])
        df = df.set_index(dates).sort_index()
        df = df.drop(columns=[date_col])
    return df


def _load_preds(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    # handle either: index saved as column or explicit date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        # assume first column is date/index, but don't overwrite its dtype
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col])
        df = df.set_index(dates).sort_index()
        df = df.drop(columns=[date_col])

    # normalize expected column names
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Expected columns y_true,y_pred in {fp}, found: {list(df.columns)}")
    return df[["y_true", "y_pred"]]


def _prob_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("p_state_")]
    if not cols:
        raise ValueError("No p_state_* columns found in oos regime probs.")
    return cols


def compute_per_regime_metrics(cfg: PerRegimeMetricsConfig) -> Path:
    probs = _load_probs(cfg.oos_probs_csv)
    preds = _load_preds(cfg.oos_predictions_csv)

    df = preds.join(probs, how="inner")
    if df.empty:
        raise ValueError("No overlapping dates between oos_predictions and oos_regime_probs.")

    if "hard_state" not in df.columns:
        # fall back to argmax if missing
        pcols = _prob_cols(df)
        df["hard_state"] = df[pcols].to_numpy().argmax(axis=1)

    if "max_prob" not in df.columns:
        pcols = _prob_cols(df)
        df["max_prob"] = df[pcols].to_numpy().max(axis=1)

    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        raise ValueError("After dropping NaNs in y_true/y_pred, no rows remain.")

    # hv_fut defined globally over OOS y_true
    y_true = df["y_true"].to_numpy(dtype=float)
    thresh = float(np.quantile(y_true, cfg.hv_fut_q))
    df["hv_fut"] = df["y_true"] >= thresh

    K = int(df["hard_state"].max()) + 1

    rows = []
    for k in range(K):
        sub = df[df["hard_state"] == k]
        n = int(len(sub))
        if n == 0:
            continue

        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)

        row = {
            "regime": int(k),
            "count": n,
            "frac": float(n / len(df)),
            "rmse": float(rmse(yt, yp)),
            "mae": float(mae(yt, yp)),
            "spearman": float(spearman_corr(yt, yp)),
            "top_decile_hit": float(top_decile_hit_rate(yt, yp)),
            "avg_conf": float(sub["max_prob"].mean()),
        }

        sub_hv = sub[sub["hv_fut"]]
        n_hv = int(len(sub_hv))
        row["hv_fut_count"] = n_hv
        row["hv_fut_frac_within_regime"] = float(n_hv / n) if n > 0 else float("nan")

        if n_hv > 0:
            yt_hv = sub_hv["y_true"].to_numpy(dtype=float)
            yp_hv = sub_hv["y_pred"].to_numpy(dtype=float)
            row["rmse_hv_fut"] = float(rmse(yt_hv, yp_hv))
            row["mae_hv_fut"] = float(mae(yt_hv, yp_hv))
        else:
            row["rmse_hv_fut"] = float("nan")
            row["mae_hv_fut"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)
    return cfg.out_csv


if __name__ == "__main__":
    # example usage:
    # python -m src.experiments.per_regime_metrics
    run_id = "results_2026-02-04T14-07-03Z"
    target = "y_absret_h1"
    model = "hmm_ridge_soft_K3_normal"

    base = Path("artifacts") / "regimes" / run_id / target / model
    cfg = PerRegimeMetricsConfig(
        oos_probs_csv=base / "oos_regime_probs.csv",
        oos_predictions_csv=base / "oos_predictions.csv",
        out_csv=base / "per_regime_metrics.csv",
        hv_fut_q=0.9,
    )
    path = compute_per_regime_metrics(cfg)
    print(f"Wrote: {path}")