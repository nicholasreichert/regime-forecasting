"""Recompute the Diebold-Mariano table from stored out-of-sample predictions.

Like :mod:`src.experiments.recompute_metrics`, this exists so that a change to a
loss definition does not require refitting the study. The tests are a pure
function of the saved prediction series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.eval.dm_test import diebold_mariano, holm_bonferroni

STUDY = Path("artifacts") / "study"
FOCAL = "regime_normal"


def _latest_run() -> Path:
    pointer = STUDY / "latest" / "run_id.txt"
    if pointer.exists():
        d = STUDY / pointer.read_text(encoding="utf-8").strip()
        if d.exists():
            return d
    runs = sorted(p for p in STUDY.glob("study_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError("no study run found")
    return runs[-1]


def main() -> Path:
    run_dir = _latest_run()
    metrics = pd.read_csv(run_dir / "metrics.csv")

    rows: List[dict] = []
    for target_dir in sorted((run_dir / "predictions").iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        prior = metrics[metrics["target"] == target]
        if prior.empty:
            continue
        h = int(prior.iloc[0]["horizon"])

        preds = {
            fp.stem: pd.read_csv(fp, index_col=0, parse_dates=True)
            for fp in sorted(target_dir.glob("*.csv"))
        }
        if FOCAL not in preds:
            continue
        focal = preds[FOCAL]

        for loss in ("mse", "qlike"):
            pvals: Dict[str, float] = {}
            staged: List[dict] = []
            for name, df in preds.items():
                if name == FOCAL:
                    continue
                common = focal.index.intersection(df.index)
                res = diebold_mariano(
                    focal.loc[common, "y_true"].to_numpy(),
                    focal.loc[common, "y_pred"].to_numpy(),
                    df.loc[common, "y_pred"].to_numpy(),
                    horizon=h,
                    loss=loss,
                )
                pvals[name] = res.p_value
                staged.append({
                    "target": target, "horizon": h, "loss": loss,
                    "model_a": FOCAL, "model_b": name,
                    "dm_stat": res.stat, "p_value": res.p_value,
                    "mean_loss_diff": res.mean_loss_diff, "favours": res.favours,
                    "n": res.n, "hac_lag": res.lag,
                })
            rejects = holm_bonferroni(pvals, alpha=0.05)
            for r in staged:
                r["reject_h0_holm_5pct"] = bool(rejects.get(r["model_b"], False))
            rows.extend(staged)

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "dm_tests.csv", index=False)
    out.to_csv(STUDY / "latest" / "dm_tests.csv", index=False)
    print(f"recomputed {len(out)} tests -> {run_dir / 'dm_tests.csv'}")
    return run_dir / "dm_tests.csv"


if __name__ == "__main__":
    main()
