from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    df = pd.read_csv(fp)
    # expect either date column or already-indexed
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        # try first column as date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
    return df


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

    # Else use returns explicitly
    if cfg.ret_col is not None and cfg.ret_col in data.columns:
        r = data[cfg.ret_col].astype(float)
        return np.log1p(r).cumsum()

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
    probs = oos[pcols].to_numpy(dtype=float)

    # align to dates that exist in data
    aligned = data.join(oos, how="inner")
    if aligned.empty:
        raise ValueError("No overlapping dates between data and OOS regime probs.")

    # rebuild probs after join to preserve alignment
    oos_aligned = aligned[pcols]
    probs_aligned = oos_aligned.to_numpy(dtype=float)
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
    labels = {ordered[0]: "Low Vol", ordered[-1]: "High Vol"}

    # Shade both panels using the SAME states (consistent boundaries)
    _shade_with_states(ax1, x, states, maxp, cfg)
    ax1.plot(x, level, lw=1.2)
    ax1.axvline(oos.index.min(), ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel(
        "log price" if (cfg.price_col and cfg.price_col in aligned.columns) else f"cum log(1+{cfg.ret_col})"
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
