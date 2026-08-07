"""Recompute the study metrics table from stored out-of-sample predictions.

The study driver saves every model's pooled ``(y_true, y_pred)`` series, so the
metrics table is a pure function of those files. When a metric definition
changes there is no need to refit anything - recompute here and the paper stays
consistent with the current code.

Selection diagnostics (chosen K, soft-gating fraction) cannot be recovered from
predictions alone and are carried over from the existing table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.oos import OOSResult, compute_metrics

STUDY = Path("artifacts") / "study"
CARRY_OVER = ["sel_K_mean", "sel_K_mode", "sel_soft_frac", "mean_experts_fitted",
              "ablation_train", "ablation_test"]


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
    old = pd.read_csv(run_dir / "metrics.csv")
    data = build_and_save_processed_dataset(load_config()).df

    rows = []
    for target_dir in sorted((run_dir / "predictions").iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        prior = old[old["target"] == target]
        if prior.empty:
            continue
        horizon = int(prior.iloc[0]["horizon"])
        family = str(prior.iloc[0]["target_family"])

        for fp in sorted(target_dir.glob("*.csv")):
            model = fp.stem
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            res = OOSResult(y_true=df["y_true"], y_pred=df["y_pred"])
            m = compute_metrics(res, data)

            row = {
                "target": target,
                "horizon": horizon,
                "target_family": family,
                "model": model,
                "model_family": "regime" if model.startswith("regime_") else "baseline",
                **m,
            }
            match = prior[prior["model"] == model]
            if not match.empty:
                for c in CARRY_OVER:
                    if c in match.columns:
                        row[c] = match.iloc[0][c]
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "metrics.csv", index=False)
    out.to_csv(STUDY / "latest" / "metrics.csv", index=False)
    print(f"recomputed {len(out)} rows -> {run_dir / 'metrics.csv'}")
    return run_dir / "metrics.csv"


if __name__ == "__main__":
    main()
