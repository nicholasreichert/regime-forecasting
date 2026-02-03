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
    alpha_max: float = 0.22

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


def _shade_segments(ax, x: pd.DatetimeIndex, probs: np.ndarray, cfg: RegimeShadingConfig) -> None:
    # shade contiguous segments w/ hard argmax state
    # alpha encodes confidence via max_prob aggregated over each segment
    states = probs.argmax(axis=1)
    maxp = probs.max(axis=1)

    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or states[i] != states[start]:
            k = int(states[start])
            color = cfg.colors[k % len(cfg.colors)]

            if cfg.use_soft_alpha:
                seg_conf = float(np.median(maxp[start:i]))  # segment confidence
                alpha = float(np.clip(seg_conf, cfg.alpha_min, cfg.alpha_max))
            else:
                alpha = float(cfg.alpha_max)

            ax.axvspan(
                x[start],
                x[i - 1],
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            start = i

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

    if "ret" in data.columns:
        r = data["ret"].astype(float)
        return r.rolling(cfg.vol_fallback_window).std()

    raise ValueError(f"Could not build vol series: missing '{cfg.vol_col}' and no 'ret' column found.")


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

    _shade_segments(ax1, x, probs_aligned, cfg)
    ax1.plot(x, level, lw=1.2)
    ax1.set_ylabel("log price" if (cfg.price_col and cfg.price_col in aligned.columns) else f"cum log(1+{cfg.ret_col})")

    _shade_segments(ax2, x, probs_aligned, cfg)
    ax2.plot(x, vol, lw=1.2)
    ax2.set_ylabel(cfg.vol_col if (cfg.vol_col and cfg.vol_col in aligned.columns) else f"ret vol ({cfg.vol_fallback_window}d)")
    ax2.set_xlabel("date")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(cfg.out_path, dpi=200)
    plt.close(fig)
    return cfg.out_path
