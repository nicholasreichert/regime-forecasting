"""Are the inferred regimes stable and separated across walk-forward folds?

This diagnostic previously fitted its own HMM on a different emission vector than
the forecasting pipeline, and labelled states with ``hmm.predict`` (Viterbi),
which conditions on the whole sequence. It therefore did not describe the model
the paper reports. It now uses exactly the pipeline's fit and its causal
filtered posteriors, so the numbers refer to the model actually being evaluated.

Two questions are asked per fold:

*separation* - do the canonical states have distinct realized-volatility levels?
*stability*  - are those levels comparable from one fold to the next?

Canonical ordering (low->high volatility, see ``src.regime.hmm``) is what makes
the second question answerable: raw Baum-Welch state indices are arbitrary, so
without relabelling "state 0" means something different in every fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.walk_forward import walk_forward_splits
from src.regime.hmm import fit_hmm_and_infer_probs


@dataclass(frozen=True)
class RegimeStabilityConfig:
    K: int = 3
    covariance_type: str = "diag"
    n_iter: int = 400
    tol: float = 1e-4
    min_covar: float = 1e-3
    seed: int = 42

    train_years: int = 6
    test_years: int = 1
    step_years: int = 1
    embargo: int = 0

    stress_feature: str = "ret_vol_20"


def compute_regime_stability(
    df: pd.DataFrame, cfg: RegimeStabilityConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.stress_feature not in df.columns:
        raise ValueError(f"Missing stress feature '{cfg.stress_feature}'")

    rows: List[dict] = []
    failed = 0

    splits = list(
        walk_forward_splits(
            df.index, cfg.train_years, cfg.test_years, cfg.step_years, embargo=cfg.embargo
        )
    )

    for i, split in enumerate(splits):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        try:
            res = fit_hmm_and_infer_probs(
                train_df=train,
                test_df=test,
                K=cfg.K,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                tol=cfg.tol,
                min_covar=cfg.min_covar,
                seed=cfg.seed,
            )
        except Exception as e:
            failed += 1
            print(f"[warn] split {i} failed: {e}")
            continue

        # Causal filtered hard states on the training window
        n = min(len(train), res.train_probs.shape[0])
        states = res.train_probs[-n:].argmax(axis=1)
        stress = train[cfg.stress_feature].to_numpy(dtype=float)[-n:]

        counts = np.array([(states == k).sum() for k in range(cfg.K)])
        if (counts == 0).any():
            failed += 1
            print(f"[warn] split {i} collapsed (empty state): counts={counts.tolist()}")
            continue

        mu = np.array([float(np.nanmean(stress[states == k])) for k in range(cfg.K)])

        row = {
            "split": f"{train.index.min().date()}_{train.index.max().date()}",
            "test_start": str(split.test_start.date()),
        }
        for k in range(cfg.K):
            row[f"mu_state_{k}"] = mu[k]
            row[f"count_state_{k}"] = int(counts[k])
        # canonical ordering means mu should already be increasing
        row["monotone"] = bool(np.all(np.diff(mu) > 0))
        row["separation_lo_hi"] = float(mu[-1] - mu[0])
        row["persistence"] = float(np.mean(np.diag(res.model.transmat_)))
        rows.append(row)

    windows = pd.DataFrame(rows)
    if len(windows) < 3:
        raise RuntimeError(f"Too few usable splits (used={len(windows)}, failed={failed})")

    summary_row = {
        "K": cfg.K,
        "splits_used": int(len(windows)),
        "splits_failed_or_collapsed": int(failed),
        "monotone_frac": float(windows["monotone"].mean()),
        "separation_mean": float(windows["separation_lo_hi"].mean()),
        "separation_std": float(windows["separation_lo_hi"].std(ddof=0)),
        "mean_self_transition": float(windows["persistence"].mean()),
    }
    for k in range(cfg.K):
        col = f"mu_state_{k}"
        summary_row[f"{col}_mean"] = float(windows[col].mean())
        # cross-fold coefficient of variation: how reproducible this state's level is
        summary_row[f"{col}_cv"] = float(windows[col].std(ddof=0) / windows[col].mean())

    return windows, pd.DataFrame([summary_row])


def main() -> None:
    cfg_all = load_config()
    df = build_and_save_processed_dataset(cfg_all).df

    out_dir = Path("artifacts") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for K in [int(k) for k in cfg_all.hmm.K_values]:
        c = RegimeStabilityConfig(
            K=K,
            covariance_type=cfg_all.hmm.covariance_type,
            n_iter=cfg_all.hmm.n_iter,
            tol=cfg_all.hmm.tol,
            min_covar=cfg_all.hmm.min_covar,
            seed=cfg_all.project.seed,
            train_years=cfg_all.evaluation.train_years,
            test_years=cfg_all.evaluation.test_years,
            step_years=cfg_all.evaluation.step_years,
        )
        windows, summary = compute_regime_stability(df, c)
        windows.to_csv(out_dir / f"regime_stability_windows_K{K}.csv", index=False)
        all_summaries.append(summary)
        print(f"\n--- K={K} ---")
        print(summary.to_string(index=False))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(out_dir / "regime_stability_summary.csv", index=False)
    (out_dir / "regime_stability_summary.tex").write_text(
        combined.to_latex(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8"
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
