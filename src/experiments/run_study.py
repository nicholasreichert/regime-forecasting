"""End-to-end study driver.

Produces every number in the paper: the model comparison table, the ablation
table, the Diebold-Mariano tests, and the per-regime breakdowns. One run of
``python -m src.experiments.run_study`` regenerates all of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.dm_test import diebold_mariano, holm_bonferroni
from src.eval.oos import (
    ABLATIONS,
    EvalSpec,
    HMMCache,
    OOSResult,
    collect_oos_baseline,
    collect_oos_regime_nested,
    compute_metrics,
)
from src.experiments.logging import utc_run_id
from src.models.baselines import (
    EWMAVolBaseline,
    GARCHBaseline,
    HARBaseline,
    RandomWalkVolBaseline,
    RidgeBaseline,
    RollingMeanBaseline,
    ZeroReturnBaseline,
)

# Backward RV window used by the random-walk baseline at each horizon.
RW_COL_FOR_HORIZON = {1: "rv_bwd_1", 5: "rv_bwd_5", 20: "rv_bwd_22"}

# The reference model that every significance test is run against.
FOCAL_MODEL = "regime_normal"


def _baseline_factories(h: int) -> Dict[str, Callable[[], object]]:
    return {
        "zero": ZeroReturnBaseline,
        "train_mean": RollingMeanBaseline,
        "rw_vol": lambda: RandomWalkVolBaseline(RW_COL_FOR_HORIZON.get(h, "rv_bwd_22")),
        "ridge": lambda: RidgeBaseline(alpha=None),
        "har": HARBaseline,
        "ewma": lambda: EWMAVolBaseline(horizon=h),
        "garch11": lambda: GARCHBaseline(horizon=h),
    }


def run_study(out_root: Path | None = None) -> Path:
    cfg = load_config()
    ds = build_and_save_processed_dataset(cfg)
    data = ds.df

    features = [c for c in data.columns if not c.startswith("y_")]

    horizons = [int(h) for h in cfg.targets.horizons]
    targets: List[tuple[str, int, str]] = []
    for h in horizons:
        targets.append((f"y_rv_h{h}", h, "realized_vol"))
    for h in horizons:
        targets.append((f"y_{cfg.targets.target_type}_h{h}", h, "point_absret"))

    run_id = utc_run_id("study")
    out_root = out_root or (Path("artifacts") / "study")
    out_dir = out_root / run_id
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type,
        n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol,
        min_covar=cfg.hmm.min_covar,
    )

    rows: List[dict] = []
    dm_rows: List[dict] = []
    regime_artifacts: List[dict] = []

    # HMM fits depend only on the split geometry and K, never on the target, so
    # a single cache is shared across every target that has the same horizon.
    cache = HMMCache(hmm_kwargs)

    for target, h, family in targets:
        print(f"\n=== {target}  (h={h}, {family}) ===", flush=True)
        ev = EvalSpec(
            train_years=cfg.evaluation.train_years,
            test_years=cfg.evaluation.test_years,
            step_years=cfg.evaluation.step_years,
            embargo=h,  # purge training rows whose targets reach into the test fold
        )
        preds: Dict[str, OOSResult] = {}

        # ---- baselines --------------------------------------------------
        for name, factory in _baseline_factories(h).items():
            try:
                res = collect_oos_baseline(data, features, target, factory, ev)
            except Exception as e:  # a baseline failing must not sink the run
                print(f"  [warn] baseline {name} failed: {e}", flush=True)
                continue
            preds[name] = res
            m = compute_metrics(res, data)
            rows.append({"target": target, "horizon": h, "target_family": family,
                         "model": name, "model_family": "baseline", **m})
            print(f"  {name:<14} rmse={m['rmse']:.6f} qlike={m['qlike']:.4f}", flush=True)

        # ---- regime-conditioned models + ablations ----------------------
        for ab in ABLATIONS:
            name = f"regime_{ab.name}"
            try:
                res = collect_oos_regime_nested(
                    df=data,
                    features=features,
                    target=target,
                    ev=ev,
                    K_values=[int(k) for k in cfg.hmm.K_values],
                    modes=["hard", "soft"],
                    ablation=ab,
                    cache=cache,
                    ridge_alpha=None,
                    min_points_per_regime=200,
                    seed=cfg.project.seed,
                )
            except Exception as e:
                print(f"  [warn] {name} failed: {e}", flush=True)
                continue
            preds[name] = res
            m = compute_metrics(res, data)
            rows.append({"target": target, "horizon": h, "target_family": family,
                         "model": name, "model_family": "regime",
                         "ablation_train": ab.train, "ablation_test": ab.test, **m})
            print(f"  {name:<20} rmse={m['rmse']:.6f} qlike={m['qlike']:.4f} "
                  f"K*={m.get('sel_K_mode')} soft={m.get('sel_soft_frac'):.2f}", flush=True)

            if ab.name == "normal" and res.regime_probs is not None:
                rdir = out_dir / "regimes" / target
                rdir.mkdir(parents=True, exist_ok=True)
                res.regime_probs.to_csv(rdir / "oos_regime_probs.csv")
                if res.hmm_info is not None:
                    np.savez(rdir / "hmm_interpretability.npz", **res.hmm_info)
                regime_artifacts.append({"target": target, "dir": str(rdir)})

        # ---- save pooled predictions ------------------------------------
        pdir = out_dir / "predictions" / target
        pdir.mkdir(parents=True, exist_ok=True)
        for name, res in preds.items():
            pd.DataFrame({"y_true": res.y_true, "y_pred": res.y_pred}).to_csv(pdir / f"{name}.csv")

        # ---- Diebold-Mariano tests --------------------------------------
        if FOCAL_MODEL in preds:
            focal = preds[FOCAL_MODEL]
            for loss in ("mse", "qlike"):
                pvals: Dict[str, float] = {}
                staged: List[dict] = []
                for name, res in preds.items():
                    if name == FOCAL_MODEL:
                        continue
                    common = focal.y_true.index.intersection(res.y_true.index)
                    dm = diebold_mariano(
                        focal.y_true.loc[common].to_numpy(),
                        focal.y_pred.loc[common].to_numpy(),
                        res.y_pred.loc[common].to_numpy(),
                        horizon=h,
                        loss=loss,
                    )
                    pvals[name] = dm.p_value
                    staged.append({"target": target, "horizon": h, "loss": loss,
                                   "model_a": FOCAL_MODEL, "model_b": name,
                                   "dm_stat": dm.stat, "p_value": dm.p_value,
                                   "mean_loss_diff": dm.mean_loss_diff,
                                   "favours": dm.favours, "n": dm.n, "hac_lag": dm.lag})
                rejects = holm_bonferroni(pvals, alpha=0.05)
                for r in staged:
                    r["reject_h0_holm_5pct"] = bool(rejects.get(r["model_b"], False))
                dm_rows.extend(staged)

        print(f"  [hmm cache] {cache.hits} hits / {cache.misses} fits", flush=True)

    # ---- persist ---------------------------------------------------------
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    dm_df = pd.DataFrame(dm_rows)
    if len(dm_df):
        dm_df.to_csv(out_dir / "dm_tests.csv", index=False)

    meta = {
        "run_id": run_id,
        "config": {
            "project": asdict(cfg.project),
            "data": asdict(cfg.data),
            "targets": asdict(cfg.targets),
            "features": asdict(cfg.features),
            "evaluation": asdict(cfg.evaluation),
            "hmm": asdict(cfg.hmm),
        },
        "n_rows": int(len(data)),
        "date_range": [str(data.index.min().date()), str(data.index.max().date())],
        "ablations": [asdict(a) for a in ABLATIONS],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(latest / "metrics.csv", index=False)
    if len(dm_df):
        dm_df.to_csv(latest / "dm_tests.csv", index=False)
    (latest / "run_id.txt").write_text(run_id, encoding="utf-8")

    print(f"\nWrote study to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_study()
