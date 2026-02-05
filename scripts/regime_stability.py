from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from src.eval.walk_forward import walk_forward_splits
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class RegimeStabilityConfig:
    processed_parquet: Path
    emission_features: List[str]
    stress_feature: str

    K: int = 3
    covariance_type: str = "full"
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 0

    train_years: int = 6
    test_years: int = 1
    step_years: int = 1

    date_col: Optional[str] = None


def _load_processed(path: Path, date_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col is None or date_col not in df.columns:
            raise ValueError("Processed parquet must have DatetimeIndex or provide date_col in config.")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    df = df.sort_index()
    return df


def compute_regime_ordered_stability(cfg: RegimeStabilityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_processed(cfg.processed_parquet, cfg.date_col)

    for c in cfg.emission_features + [cfg.stress_feature]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in processed parquet.")

    X_all = df[cfg.emission_features].astype(float)

    rows = []
    failed = 0

    for i, split in enumerate(walk_forward_splits(df.index, cfg.train_years, cfg.test_years, cfg.step_years)):
        train_idx = split.train_idx
        train_df = df.loc[train_idx]
        X_train = X_all.loc[train_idx].to_numpy(dtype=float)
        stress_train = train_df[cfg.stress_feature].to_numpy(dtype=float)

        # standardize emissions within each training window
        X_train = StandardScaler().fit_transform(X_train)

        hmm = GaussianHMM(
            n_components=cfg.K,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tol,
            random_state=cfg.random_state,
            # stabilizer (helps avoid singularities even for diag sometimes)
            min_covar=1e-4,
        )

        try:
            hmm.fit(X_train)
            states_train = hmm.predict(X_train)
        except Exception as e:
            failed += 1
            print(f"[warn] split {i} failed: {e}")
            continue

        # mean stress per (raw) state id
        K = cfg.K
        mu = np.full(K, np.nan, dtype=float)
        counts = np.zeros(K, dtype=int)
        for k in range(K):
            mask = (states_train == k)
            counts[k] = int(mask.sum())
            if counts[k] > 0:
                mu[k] = float(np.nanmean(stress_train[mask]))

        # If any state is empty, the HMM effectively collapsed; skip this split
        if np.isnan(mu).any():
            failed += 1
            print(f"[warn] split {i} collapsed (empty state). counts={counts.tolist()}")
            continue

        # ordered regimes within this split (low -> high stress)
        order = np.argsort(mu)
        mu_low, mu_mid, mu_high = mu[order[0]], mu[order[1]], mu[order[2]]

        rows.append(
            {
                "split": f"{train_df.index.min().date()}_{train_df.index.max().date()}",
                "raw_state_low": int(order[0]),
                "raw_state_mid": int(order[1]),
                "raw_state_high": int(order[2]),
                "mu_low": float(mu_low),
                "mu_mid": float(mu_mid),
                "mu_high": float(mu_high),
                "gap_low_mid": float(mu_mid - mu_low),
                "gap_mid_high": float(mu_high - mu_mid),
                "gap_low_high": float(mu_high - mu_low),
                "count_low": int(counts[order[0]]),
                "count_mid": int(counts[order[1]]),
                "count_high": int(counts[order[2]]),
            }
        )

    windows = pd.DataFrame(rows)

    if len(windows) < 3:
        raise RuntimeError(
            f"Not enough successful splits to assess stability (used={len(windows)}, failed={failed})."
        )

    summary = pd.DataFrame(
        [
            {
                "splits_used": int(len(windows)),
                "splits_failed_or_collapsed": int(failed),
                "mu_low_mean": float(windows["mu_low"].mean()),
                "mu_mid_mean": float(windows["mu_mid"].mean()),
                "mu_high_mean": float(windows["mu_high"].mean()),
                "gap_low_mid_mean": float(windows["gap_low_mid"].mean()),
                "gap_low_mid_std": float(windows["gap_low_mid"].std(ddof=0)),
                "gap_mid_high_mean": float(windows["gap_mid_high"].mean()),
                "gap_mid_high_std": float(windows["gap_mid_high"].std(ddof=0)),
                "gap_low_high_mean": float(windows["gap_low_high"].mean()),
                "gap_low_high_std": float(windows["gap_low_high"].std(ddof=0)),
                "p_gap_low_mid_pos": float((windows["gap_low_mid"] > 0).mean()),
                "p_gap_mid_high_pos": float((windows["gap_mid_high"] > 0).mean()),
            }
        ]
    )

    return windows, summary

def main():
    import yaml

    cfg_yaml = yaml.safe_load(Path("configs/final.yaml").read_text(encoding="utf-8"))

    c = RegimeStabilityConfig(
        processed_parquet=Path(cfg_yaml["data"]["processed_parquet"]),
        date_col=cfg_yaml["data"].get("date_col", None),
        emission_features=cfg_yaml["regime"]["emission_features"],
        stress_feature=cfg_yaml["regime"]["stress_feature"],
        K=int(cfg_yaml["regime"]["K"]),
        covariance_type=str(cfg_yaml["regime"]["hmm_covariance_type"]),
        n_iter=int(cfg_yaml["regime"]["hmm_n_iter"]),
        tol=float(cfg_yaml["regime"]["hmm_tol"]),
        random_state=int(cfg_yaml["regime"]["hmm_random_state"]),
        train_years=int(cfg_yaml["evaluation"]["train_years"]),
        test_years=int(cfg_yaml["evaluation"]["test_years"]),
        step_years=int(cfg_yaml["evaluation"]["step_years"]),
    )

    windows, summary = compute_regime_ordered_stability(c)

    out_dir = Path("artifacts") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    win_csv = out_dir / "regime_stability_windows.csv"
    sum_csv = out_dir / "regime_stability_summary.csv"
    sum_tex = out_dir / "regime_stability_summary.tex"

    windows.to_csv(win_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    sum_tex.write_text(summary.to_latex(index=False, float_format=lambda x: f"{x:.6f}"), encoding="utf-8")

    print(f"Wrote {win_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {sum_tex}")
    print(summary)


if __name__ == "__main__":
    main()
