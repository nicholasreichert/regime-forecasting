from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RegimeSummary:
    per_regime: pd.DataFrame
    transition_matrix: np.ndarray

def summarize_regimes(df: pd.DataFrame, probs: np.ndarray) -> RegimeSummary:
    """
    creates a summary table with:
    mean return per regime
    volatility per regime
    average abs return per regime
    counts of points assigned to regime
    """
    K = probs.shape[1]
    hard = probs.argmax(axis=1)

    tmp = df.copy()
    tmp = tmp.iloc[-len(hard):].copy()
    tmp["regime"] = hard

    rows = []
    for k in range(K):
        sub = tmp[tmp["regime"] == k]
        rows.append(
            {
                "regime": k,
                "count": len(sub),
                "mean_ret": float(sub["ret_1d"].mean()),
                "vol_ret": float(sub["ret_1d"].std(ddof=0)),
                "means_abs_ret": float(sub["ret_1d"].abs().mean()),
            }
        )
    
    per_regime = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)

    trans = np.zeros((K, K), dtype=float)
    for a, b in zip(hard[:-1], hard[1:]):
        trans[a, b] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums != 0)

    return RegimeSummary(per_regime=per_regime, transition_matrix=trans)