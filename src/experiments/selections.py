from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _pick_best(
    g: pd.DataFrame,
    primary: str,
    secondary: Optional[str] = None,
) -> pd.Series:
    """
    Pick best row from group g by lowest primary metric, then secondary if provided.
    Rows with NaN in primary are dropped.
    """
    gg = g.copy()
    if primary not in gg.columns:
        raise KeyError(f"Missing metric column '{primary}'")

    gg[primary] = gg[primary].apply(_safe_float)
    gg = gg[~gg[primary].isna()]
    if gg.empty:
        # No valid rows
        return pd.Series(dtype=object)

    sort_cols = [primary]
    if secondary and secondary in gg.columns:
        gg[secondary] = gg[secondary].apply(_safe_float)
        sort_cols.append(secondary)

    # Stable sort for reproducibility
    gg = gg.sort_values(sort_cols, ascending=True, kind="mergesort")
    return gg.iloc[0]


def select_best_models(results: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a tidy 'best models' table with one row per target/horizon for:
      - best_overall: lowest rmse (tie-break mae)
      - best_hv_fut: lowest rmse_hv_fut (tie-break mae_hv_fut), if available
    """
    required_cols = {"target", "horizon", "model", "family", "params", "rmse", "mae"}
    missing = required_cols - set(results.columns)
    if missing:
        raise KeyError(f"Results missing required columns: {sorted(missing)}")

    has_hv_fut = ("rmse_hv_fut" in results.columns) and ("mae_hv_fut" in results.columns)

    rows = []
    for (target, horizon), g in results.groupby(["target", "horizon"], sort=True):
        best_overall = _pick_best(g, primary="rmse", secondary="mae")
        if best_overall.empty:
            continue

        base = {
            "target": target,
            "horizon": int(horizon),
            "best_overall_model": best_overall.get("model"),
            "best_overall_family": best_overall.get("family"),
            "best_overall_rmse": _safe_float(best_overall.get("rmse")),
            "best_overall_mae": _safe_float(best_overall.get("mae")),
            "best_overall_params": best_overall.get("params"),
        }

        if has_hv_fut:
            best_hv_fut = _pick_best(g, primary="rmse_hv_fut", secondary="mae_hv_fut")
            # Some targets may not have hv_fut populated (e.g., return targets)
            if not best_hv_fut.empty:
                base.update(
                    {
                        "best_hv_fut_model": best_hv_fut.get("model"),
                        "best_hv_fut_family": best_hv_fut.get("family"),
                        "best_hv_fut_rmse": _safe_float(best_hv_fut.get("rmse_hv_fut")),
                        "best_hv_fut_mae": _safe_float(best_hv_fut.get("mae_hv_fut")),
                        "best_hv_fut_params": best_hv_fut.get("params"),
                    }
                )
            else:
                base.update(
                    {
                        "best_hv_fut_model": np.nan,
                        "best_hv_fut_family": np.nan,
                        "best_hv_fut_rmse": np.nan,
                        "best_hv_fut_mae": np.nan,
                        "best_hv_fut_params": np.nan,
                    }
                )

        rows.append(base)

    return pd.DataFrame(rows).sort_values(["target", "horizon"], kind="mergesort").reset_index(drop=True)


def save_best_models(best_df: pd.DataFrame, out_dir: Path, run_id: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_path = out_dir / f"{run_id}_best_models.csv"
    best_df.to_csv(run_path, index=False)

    latest_path = out_dir / "best_models_latest.csv"
    best_df.to_csv(latest_path, index=False)

    return run_path, latest_path
