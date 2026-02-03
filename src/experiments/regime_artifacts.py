from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def collect_oos_regime_probs(
        out_dir: Path,
        all_test_dates: list[pd.DatetimeIndex],
        all_filtered_probs: list[np.ndarray],
):
    # concatenate test-window regime probs across walk-forward splits

    rows = []
    for dates, probs in zip(all_test_dates, all_filtered_probs):
        for d, p in zip(dates, probs):
            row = {"date": d}
            for k in range(p.shape[0]):
                row[f"p_state_{k}"] = p[k]
            row["hard_state"] = int(np.argmax(p))
            row["max_prob"] = float(np.max(p))
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("date")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "oos_regime_probs.csv", index=False)
    return df

def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("hard_state")
    summary = grp.agg(
        count = ("hard_state", "size"),
        mean_confidence=("max_prob", "mean"),
    ) 
    return summary