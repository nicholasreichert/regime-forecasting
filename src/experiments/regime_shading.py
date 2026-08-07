from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeShadingConfig:
    # input artifacts
    oos_probs_csv: Path
    # output
    out_path: Path
    
    # explicit ret_col to fix error
    ret_col: Optional[str] = "ret_1d"

    # which column to use for price/level plot
    # if None, we fall back to cum log returns from 'ret' if available
    price_col: Optional[str] = "close"

    # optional vol series to plot (preferred: your engineered realized vol feature)
    vol_col: Optional[str] = "ret_vol_20"
    vol_fallback_window: int = 20  # if vol_col missing, compute rolling std(ret)

    # shading behavior
    use_soft_alpha: bool = True          # alpha scaled by max prob
    alpha_min: float = 0.05
    alpha_max: float = 0.18

    # colors (repeat if K > len(colors))
    colors: Sequence[str] = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")


def _load_oos_probs(fp: Path) -> pd.DataFrame:
    """Load an OOS regime-probability CSV, whatever the date column is called.

    Different writers in this repo emit the index as ``date`` or ``Date``, so
    match case-insensitively and otherwise fall back to the first column.
    """
    df = pd.read_csv(fp)

    date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), df.columns[0])
    idx = pd.to_datetime(df[date_col])
    df = df.drop(columns=[date_col])
    df.index = pd.DatetimeIndex(idx)
    return df.sort_index()


def _prob_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("p_state_")]
    if not cols:
        raise ValueError("No p_state_* columns found in OOS probs CSV.")
    return cols


def _shade_segments(ax, x: pd.DatetimeIndex, probs: np.ndarray, cfg: RegimeShadingConfig) -> np.ndarray:
    raw_states = probs.argmax(axis=1)
    states = _merge_short_segments(raw_states, probs, min_len=40)  # if you added merging
    maxp = probs.max(axis=1)

    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or states[i] != states[start]:
            k = int(states[start])
            color = cfg.colors[k % len(cfg.colors)]

            seg_conf = float(np.mean(maxp[start:i]))
            alpha = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * (seg_conf ** 2)

            ax.axvspan(x[start], x[i - 1], color=color, alpha=alpha, linewidth=0)
            start = i

    return states


def _compute_level_series(data: pd.DataFrame, cfg: RegimeShadingConfig) -> pd.Series:
    # If price exists, prefer it
    if cfg.price_col is not None and cfg.price_col in data.columns:
        px = data[cfg.price_col].astype(float)
        return np.log(px)

    # Else use returns explicitly. ret_1d is already a log return, so the
    # cumulative log level is its plain cumulative sum; applying log1p first
    # would treat it as a simple return and distort the level.
    if cfg.ret_col is not None and cfg.ret_col in data.columns:
        r = data[cfg.ret_col].astype(float)
        return r.cumsum()

    raise ValueError(
        f"Could not build level series: missing price_col='{cfg.price_col}' and ret_col='{cfg.ret_col}'."
    )


def _compute_vol_series(data: pd.DataFrame, cfg: RegimeShadingConfig) -> pd.Series:
    if cfg.vol_col is not None and cfg.vol_col in data.columns:
        return data[cfg.vol_col].astype(float)

    if cfg.ret_col is not None and cfg.ret_col in data.columns:
        r = data[cfg.ret_col].astype(float)
        return r.rolling(cfg.vol_fallback_window).std()

    raise ValueError(f"Could not build vol series: missing '{cfg.vol_col}' and no 'ret' column found.")

def _merge_short_segments(states: np.ndarray, probs: np.ndarray, min_len: int) -> np.ndarray:
    # merge segments shorter than min_len into neighboring segments
    # using highest mean prob
    out = states.copy()
    n = len(states)

    start = 0
    while start < n:
        end = start + 1
        while end < n and states[end] == states[start]:
            end += 1

        seg_len = end - start
        if seg_len < min_len:
            left = start - 1
            right = end

            candidates = []
            if left >= 0:
                candidates.append(out[left])
            if right < n:
                candidates.append(out[right])

            if candidates:
                # choose candidate with highest mean prob over this segment
                best = max(
                    candidates,
                    key=lambda k: probs[start:end, k].mean()
                )
                out[start:end] = best

        start = end

    return out

def _shade_with_states(ax, x: pd.DatetimeIndex, states: np.ndarray, maxp: np.ndarray, cfg: RegimeShadingConfig) -> None:
    # shade contiguous segments given precomputed hard states

    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or states[i] != states[start]:
            k = int(states[start])
            color = cfg.colors[k % len(cfg.colors)]

            seg_conf = float(np.mean(maxp[start:i]))
            alpha = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * (seg_conf ** 2)

            ax.axvspan(x[start], x[i - 1], color=color, alpha=alpha, linewidth=0)
            start = i


def make_regime_shading_plot(
    data: pd.DataFrame,
    cfg: RegimeShadingConfig,
    title: Optional[str] = None,
) -> Path:
    """
    data: processed dataset df (must include the same date index used for walk-forward)
    cfg.oos_probs_csv: artifact output from evaluate_hmm_regime_ridge aggregation
    """
    oos = _load_oos_probs(cfg.oos_probs_csv)
    pcols = _prob_cols(oos)

    # align to dates that exist in data
    aligned = data.join(oos, how="inner")
    if aligned.empty:
        raise ValueError("No overlapping dates between data and OOS regime probs.")

    # Rebuild probs after the join to preserve alignment. When K is selected per
    # fold the concatenated table is ragged (a K=2 fold has no p_state_2), so
    # missing entries mean "zero mass on a state this fold did not have"; rows
    # with no mass at all are dropped rather than allowed to poison the shading.
    probs_aligned = aligned[pcols].to_numpy(dtype=float)
    probs_aligned = np.nan_to_num(probs_aligned, nan=0.0)

    keep = probs_aligned.sum(axis=1) > 0
    if not keep.all():
        aligned = aligned.loc[keep]
        probs_aligned = probs_aligned[keep]
    if len(aligned) == 0:
        raise ValueError("No usable rows with regime mass after alignment.")

    x = aligned.index

    level = _compute_level_series(aligned, cfg)
    vol = _compute_vol_series(aligned, cfg)


    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Compute hard states once (with merging), and confidence (max prob) for alpha
    raw_states = probs_aligned.argmax(axis=1)
    states = _merge_short_segments(raw_states, probs_aligned, min_len=40)
    maxp = probs_aligned.max(axis=1)

    # Determine regime ordering by average volatility (for semantic labels)
    K = probs_aligned.shape[1]
    regime_vol = {}
    for k in range(K):
        mask = (states == k)
        regime_vol[k] = float(np.nanmean(vol[mask])) if mask.any() else float("inf")

    ordered = sorted(regime_vol, key=regime_vol.get)
    if len(ordered) == 1:
        labels = {ordered[0]: "Single regime"}
    elif len(ordered) == 2:
        labels = {ordered[0]: "Low vol", ordered[1]: "High vol"}
    else:
        labels = {ordered[0]: "Low vol", ordered[-1]: "High vol"}
        for i, k in enumerate(ordered[1:-1], start=1):
            labels[k] = "Mid vol" if len(ordered) == 3 else f"Mid vol {i}"

    # Shade both panels using the SAME states (consistent boundaries)
    _shade_with_states(ax1, x, states, maxp, cfg)
    ax1.plot(x, level, lw=1.2)
    ax1.axvline(oos.index.min(), ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel(
        "log price" if (cfg.price_col and cfg.price_col in aligned.columns) else "cumulative log return"
    )

    _shade_with_states(ax2, x, states, maxp, cfg)
    ax2.plot(x, vol, lw=1.2)
    ax2.set_ylabel(
        cfg.vol_col if (cfg.vol_col and cfg.vol_col in aligned.columns) else f"ret vol ({cfg.vol_fallback_window}d)"
    )
    ax2.set_xlabel("date")

    # Legend (show at least Low/High vol regimes)
    from matplotlib.patches import Patch
    handles = [
        Patch(color=cfg.colors[k % len(cfg.colors)], label=labels.get(k, f"Regime {k}"))
        for k in ordered
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)


    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(cfg.out_path, dpi=200)
    plt.close(fig)
    return cfg.out_path
