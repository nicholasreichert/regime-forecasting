"""How much does the result depend on the HMM's random initialization?

Baum-Welch maximises a non-convex likelihood, so the fitted regimes - and every
downstream forecast - are a function of the random seed. A single-seed result is
therefore not a property of the method, it is one draw from a distribution.

This script re-runs the full nested-selection pipeline across seeds and reports
the spread of out-of-sample RMSE, alongside the pooled-ridge baseline (which is
deterministic and so appears as a flat line). If the seed-to-seed spread is
comparable to the gap between the regime model and the baseline, no single-seed
comparison between them can be informative.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.metrics import rmse
from src.eval.oos import (
    ABLATION_BY_NAME,
    EvalSpec,
    HMMCache,
    collect_oos_baseline,
    collect_oos_regime_nested,
    compute_metrics,
)
from src.models.baselines import RidgeBaseline

SEEDS = list(range(10))


def main(out_dir: Path | None = None) -> Path:
    cfg = load_config()
    df = build_and_save_processed_dataset(cfg).df
    features = [c for c in df.columns if not c.startswith("y_")]

    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type,
        n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol,
        min_covar=cfg.hmm.min_covar,
    )

    rows: List[dict] = []
    for h in [int(x) for x in cfg.targets.horizons]:
        target = f"y_rv_h{h}"
        ev = EvalSpec(6, 1, 1, embargo=h)

        base = collect_oos_baseline(df, features, target, lambda: RidgeBaseline(alpha=None), ev)
        base_rmse = rmse(base.y_true.to_numpy(), base.y_pred.to_numpy())

        for seed in SEEDS:
            cache = HMMCache(hmm_kwargs)
            res = collect_oos_regime_nested(
                df=df,
                features=features,
                target=target,
                ev=ev,
                K_values=[int(k) for k in cfg.hmm.K_values],
                modes=["hard", "soft"],
                ablation=ABLATION_BY_NAME["normal"],
                cache=cache,
                seed=seed,
            )
            m = compute_metrics(res, df)
            rows.append({
                "horizon": h,
                "target": target,
                "seed": seed,
                "rmse_regime": m["rmse"],
                "qlike_regime": m["qlike"],
                "rmse_pooled_ridge": base_rmse,
                "improvement_pct": 100.0 * (base_rmse - m["rmse"]) / base_rmse,
                "sel_K_mode": m.get("sel_K_mode"),
                "sel_soft_frac": m.get("sel_soft_frac"),
            })
            print(
                f"h={h} seed={seed} rmse={m['rmse']:.6f} "
                f"({rows[-1]['improvement_pct']:+.2f}% vs ridge) K*={m.get('sel_K_mode')}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out_dir = out_dir or (Path("artifacts") / "study" / "seed_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "seed_robustness.csv", index=False)

    summary = (
        out.groupby("horizon")
        .agg(
            rmse_mean=("rmse_regime", "mean"),
            rmse_std=("rmse_regime", "std"),
            rmse_min=("rmse_regime", "min"),
            rmse_max=("rmse_regime", "max"),
            improvement_mean=("improvement_pct", "mean"),
            improvement_min=("improvement_pct", "min"),
            improvement_max=("improvement_pct", "max"),
            baseline=("rmse_pooled_ridge", "first"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "seed_robustness_summary.csv", index=False)
    print("\n", summary.to_string(index=False))
    print(f"\nWrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
