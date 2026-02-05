from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.eval.metrics import mae, rmse, spearman_corr, top_decile_hit_rate
from src.eval.walk_forward import walk_forward_splits
from src.models.baselines import RidgeBaseline


@dataclass(frozen=True)
class PerRegimeCompareConfig:
    processed_parquet: Path          # processed dataset parquet containing features + targets
    oos_probs_csv: Path              # .../oos_regime_probs.csv from the *best* HMM model (normal)
    hmm_oos_predictions_csv: Path    # .../oos_predictions.csv from that same HMM model
    target: str                      # e.g., y_absret_h1

    # walk-forward evaluation settings (must match training used for HMM + ridge)
    train_years: int = 6
    test_years: int = 1
    step_years: int = 1

    ridge_alpha: float = 1.0

    # outputs
    out_compare_csv: Optional[Path] = None          # per-regime ridge vs HMM comparison
    out_pubready_csv: Optional[Path] = None         # compact table for paper
    out_latex: Optional[Path] = None                # LaTeX table snippet
    out_plot_path: Optional[Path] = None            # diagnostic bar chart (hv_fut concentration)

    # define "future stress" as top q of y_true over the full OOS period (global)
    hv_fut_q: float = 0.9


def _load_probs(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col])
        df = df.set_index(dates).sort_index()
        df = df.drop(columns=[date_col])
    return df


def _load_preds(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col])
        df = df.set_index(dates).sort_index()
        df = df.drop(columns=[date_col])
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Expected columns y_true,y_pred in {fp}, found: {list(df.columns)}")
    return df[["y_true", "y_pred"]]


def _prob_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("p_state_")]
    if not cols:
        raise ValueError("No p_state_* columns found in oos regime probs.")
    return cols


def compute_ridge_oos_predictions(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    train_years: int,
    test_years: int,
    step_years: int,
    alpha: float = 1.0,
) -> pd.DataFrame:
    model = RidgeBaseline(alpha=float(alpha))

    y_true_all: list[pd.Series] = []
    y_pred_all: list[pd.Series] = []

    for split in walk_forward_splits(df.index, train_years, test_years, step_years):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=X_test.index)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true = pd.concat(y_true_all).sort_index()
    y_pred = pd.concat(y_pred_all).sort_index()
    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    return out


def _regime_labels(K: int) -> dict[int, str]:
    if K == 2:
        return {0: "low", 1: "high"}
    if K == 3:
        return {0: "low", 1: "transition", 2: "high"}
    # generic fallback
    labels = {0: "low", K - 1: "high"}
    for k in range(1, K - 1):
        labels[k] = f"mid{k}"
    return labels


def compute_per_regime_comparison(cfg: PerRegimeCompareConfig) -> Path:
    data = pd.read_parquet(cfg.processed_parquet, engine='fastparquet')
    if not isinstance(data.index, pd.DatetimeIndex):

        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.set_index("date")
        else:
            raise ValueError("Processed parquet must have a DatetimeIndex or a 'date' column.")

    features = [c for c in data.columns if not c.startswith("y_")]

    # ridge OOS predictions
    ridge_oos = compute_ridge_oos_predictions(
        df=data,
        features=features,
        target=cfg.target,
        train_years=cfg.train_years,
        test_years=cfg.test_years,
        step_years=cfg.step_years,
        alpha=cfg.ridge_alpha,
    ).rename(columns={"y_pred": "y_pred_ridge"})

    # Load HMM preds + probs
    hmm_preds = _load_preds(cfg.hmm_oos_predictions_csv).rename(columns={"y_pred": "y_pred_hmm"})
    probs = _load_probs(cfg.oos_probs_csv)

    merged = hmm_preds.join(probs, how="inner").join(ridge_oos[["y_pred_ridge"]], how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between HMM preds, regime probs, and ridge OOS preds.")

    if "hard_state" not in merged.columns:
        pcols = _prob_cols(merged)
        merged["hard_state"] = merged[pcols].to_numpy().argmax(axis=1)

    # hv_fut defined globally over OOS y_true
    y_true = merged["y_true"].to_numpy(dtype=float)
    thresh = float(np.quantile(y_true, cfg.hv_fut_q))
    merged["hv_fut"] = merged["y_true"] >= thresh

    K = int(merged["hard_state"].max()) + 1
    labels = _regime_labels(K)

    rows = []
    for k in range(K):
        sub = merged[merged["hard_state"] == k]
        if len(sub) == 0:
            continue

        yt = sub["y_true"].to_numpy(dtype=float)
        yp_r = sub["y_pred_ridge"].to_numpy(dtype=float)
        yp_h = sub["y_pred_hmm"].to_numpy(dtype=float)

        sub_hv = sub[sub["hv_fut"]]
        yt_hv = sub_hv["y_true"].to_numpy(dtype=float)
        yp_r_hv = sub_hv["y_pred_ridge"].to_numpy(dtype=float)
        yp_h_hv = sub_hv["y_pred_hmm"].to_numpy(dtype=float)

        n = int(len(sub))
        n_hv = int(len(sub_hv))

        row = {
            "regime": int(k),
            "regime_label": labels.get(int(k), str(int(k))),
            "count": n,
            "frac": float(n / len(merged)),
            "hv_fut_count": n_hv,
            "hv_fut_frac_within_regime": float(n_hv / n) if n else float("nan"),
            "hv_fut_frac_of_all_hv_fut": float(n_hv / max(int(merged["hv_fut"].sum()), 1)),
            # ridge metrics
            "ridge_rmse": float(rmse(yt, yp_r)),
            "ridge_mae": float(mae(yt, yp_r)),
            "ridge_spearman": float(spearman_corr(yt, yp_r)),
            "ridge_top_decile_hit": float(top_decile_hit_rate(yt, yp_r)),
            "ridge_rmse_hv_fut": float(rmse(yt_hv, yp_r_hv)) if n_hv else float("nan"),
            "ridge_mae_hv_fut": float(mae(yt_hv, yp_r_hv)) if n_hv else float("nan"),
            # HMM metrics
            "hmm_rmse": float(rmse(yt, yp_h)),
            "hmm_mae": float(mae(yt, yp_h)),
            "hmm_spearman": float(spearman_corr(yt, yp_h)),
            "hmm_top_decile_hit": float(top_decile_hit_rate(yt, yp_h)),
            "hmm_rmse_hv_fut": float(rmse(yt_hv, yp_h_hv)) if n_hv else float("nan"),
            "hmm_mae_hv_fut": float(mae(yt_hv, yp_h_hv)) if n_hv else float("nan"),
        }

        # deltas (HMM - Ridge): positive => improvement by HMM
        row.update(
            {
                "delta_rmse": float(row["ridge_rmse"] - row["hmm_rmse"]),
                "delta_mae": float(row["ridge_mae"] - row["hmm_mae"]),
                "delta_rmse_hv_fut": float(row["ridge_rmse_hv_fut"] - row["hmm_rmse_hv_fut"]),
                "delta_mae_hv_fut": float(row["ridge_mae_hv_fut"] - row["hmm_mae_hv_fut"]),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)

    # outputs
    base_dir = cfg.out_compare_csv.parent if cfg.out_compare_csv else cfg.oos_probs_csv.parent
    out_csv = cfg.out_compare_csv or (base_dir / "per_regime_compare_ridge_vs_hmm.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # table
    pub = out[["regime", "regime_label", "frac", "hv_fut_frac_of_all_hv_fut", "ridge_rmse_hv_fut", "hmm_rmse_hv_fut", "delta_rmse_hv_fut"]].copy()
    pub = pub.rename(
        columns={
            "frac": "time_frac",
            "hv_fut_frac_of_all_hv_fut": "hv_fut_concentration",
            "delta_rmse_hv_fut": "improvement_rmse_hv_fut",
        }
    )
    pub_csv = cfg.out_pubready_csv or (base_dir / "per_regime_pubready_table.csv")
    pub.to_csv(pub_csv, index=False)

    # latex config if wanted
    latex_path = cfg.out_latex or (base_dir / "per_regime_pubready_table.tex")
    latex_path.write_text(
        pub.to_latex(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        encoding="utf-8",
    )

    # opt diagnostic plot: hv_fut concentration by regime
    if cfg.out_plot_path is not None:
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(pub["regime_label"], pub["hv_fut_concentration"].astype(float))
            ax.set_title("Future-stress (hv_fut) concentration by regime")
            ax.set_ylabel("Fraction of all hv_fut events")
            ax.set_xlabel("Regime")
            fig.tight_layout()
            cfg.out_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(cfg.out_plot_path, dpi=200)
            plt.close(fig)
        except Exception:
            pass

    return out_csv


if __name__ == "__main__":
    run_id = "results_2026-02-04T14-07-03Z"
    target = "y_absret_h1"
    model = "hmm_ridge_soft_K3_normal"

    base = Path("artifacts") / "regimes" / run_id / target / model
    cfg = PerRegimeCompareConfig(
        processed_parquet=Path("data/processed/SPY_absret_h1-5-20_features.parquet"),
        oos_probs_csv=base / "oos_regime_probs.csv",
        hmm_oos_predictions_csv=base / "oos_predictions.csv",
        target=target,
        out_plot_path=base / "plots" / "hv_fut_concentration_by_regime.png",
    )
    out_path = compute_per_regime_comparison(cfg)
    print(f"Wrote: {out_path}")
