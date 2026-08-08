"""Does estimating the states jointly with the forecast fix the diagnosed failure?

The simulation study showed that the two-stage pipeline -- fit an HMM to
returns, then condition a regressor on the inferred state -- recovers states
that differ in volatility *level* and largely misses states that differ in
volatility *dynamics*, because Baum-Welch maximises the likelihood of the
emissions rather than of the target. The differing-dynamics world is exactly
where it fails.

A Markov-switching regression estimates the states from the predictive density
instead. If the diagnosis is right, it should recover that world. This script
runs both on the same synthetic data and on real returns.

The honest prediction is that joint estimation fixes the synthetic case and
still does not help on real data, because the redundancy problem is separate:
there is no residual regime structure in SPY for a better estimator to find.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.features import build_feature_table  # noqa: E402
from src.data.make_dataset import load_or_download_equity_data  # noqa: E402
from src.eval.dm_test import diebold_mariano  # noqa: E402
from src.eval.metrics import rmse  # noqa: E402
from src.eval.oos import (  # noqa: E402
    ABLATION_BY_NAME,
    EvalSpec,
    HMMCache,
    collect_oos_markov_switching,
    collect_oos_regime_nested,
)
from src.experiments.simulation_study import (  # noqa: E402
    _TRUE_STATE,
    DGPS,
    N_OBS,
    _frame_from_returns,
)

HORIZONS = [1, 5, 20]
CASES = ["ms_slope", "ms_level", "garch11", "SPY"]


def _dataset(case: str, rep: int):
    cfg = load_config()
    if case == "SPY":
        raw = load_or_download_equity_data(cfg)
        return build_feature_table(cfg, raw.df), None
    rng = np.random.default_rng(1000 * rep + abs(hash(case)) % 1000)
    returns = DGPS[case](N_OBS, rng)
    true_state = _TRUE_STATE.get("last")
    raw = _frame_from_returns(returns, rng)
    data = build_feature_table(cfg, raw)
    if true_state is not None:
        true_state = np.asarray(true_state)[raw.index.get_indexer(data.index)]
    return data, true_state


def run_case(case: str, rep: int) -> List[dict]:
    cfg = load_config()
    data, _ = _dataset(case, rep)
    features = [c for c in data.columns if not c.startswith("y_")]
    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type, n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol, min_covar=cfg.hmm.min_covar,
    )
    cache = HMMCache(hmm_kwargs)

    rows: List[dict] = []
    for h in HORIZONS:
        target = f"y_rv_h{h}"
        ev = EvalSpec(cfg.evaluation.train_years, cfg.evaluation.test_years,
                      cfg.evaluation.step_years, embargo=h)
        try:
            pooled = collect_oos_regime_nested(
                df=data, features=features, target=target, ev=ev,
                K_values=[2, 3], modes=["hard", "soft"],
                ablation=ABLATION_BY_NAME["single"], cache=cache, seed=cfg.project.seed,
            )
            twostage = collect_oos_regime_nested(
                df=data, features=features, target=target, ev=ev,
                K_values=[2, 3], modes=["hard", "soft"],
                ablation=ABLATION_BY_NAME["normal"], cache=cache, seed=cfg.project.seed,
            )
            joint = collect_oos_markov_switching(
                df=data, features=features, target=target, ev=ev,
                K_values=[2, 3], seed=cfg.project.seed,
            )
        except Exception as e:
            rows.append({"case": case, "rep": rep, "horizon": h,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        idx = pooled.y_true.index.intersection(twostage.y_true.index).intersection(joint.y_true.index)
        y = pooled.y_true.loc[idx].to_numpy()
        r_pool = rmse(y, pooled.y_pred.loc[idx].to_numpy())
        r_two = rmse(y, twostage.y_pred.loc[idx].to_numpy())
        r_joint = rmse(y, joint.y_pred.loc[idx].to_numpy())

        dm = diebold_mariano(y, joint.y_pred.loc[idx].to_numpy(),
                             pooled.y_pred.loc[idx].to_numpy(), horizon=h, loss="mse")

        rows.append({
            "case": case, "rep": rep, "horizon": h,
            "rmse_pooled": r_pool, "rmse_twostage": r_two, "rmse_joint": r_joint,
            "twostage_vs_pooled_pct": 100.0 * (r_pool - r_two) / r_pool,
            "joint_vs_pooled_pct": 100.0 * (r_pool - r_joint) / r_pool,
            "joint_dm_p": dm.p_value,
        })
    return rows


def main(n_reps: int = 4, workers: int = 12) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(c, r) for c in CASES for r in (range(n_reps) if c != "SPY" else [0])]
    print(f"=== {len(jobs)} runs on {workers} workers ===", flush=True)

    all_rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_case, c, r): (c, r) for c, r in jobs}
        for fut in as_completed(futs):
            c, r = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[fail] {c} rep{r}: {type(e).__name__}: {e}", flush=True)
                continue
            all_rows.extend(rows)
            ok = [x for x in rows if "error" not in x]
            if ok:
                msg = "  ".join(
                    f"h={x['horizon']}: two-stage {x['twostage_vs_pooled_pct']:+.2f}% "
                    f"joint {x['joint_vs_pooled_pct']:+.2f}%" for x in ok)
                print(f"[done] {c:<9} rep{r}  {msg}", flush=True)

    df = pd.DataFrame(all_rows)
    out = Path("artifacts") / "study" / "joint_vs_twostage"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "joint_vs_twostage.csv", index=False)
    print(f"\nWrote {out / 'joint_vs_twostage.csv'}")
    return out


if __name__ == "__main__":
    main()
