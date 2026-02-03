from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RegimeShadingconfig:
    artifact_dir: Path
    out_dir: Path
    date_col: str = "date"
    price_col: str = "close"
    rv_window: int = 20
    use_soft: bool = True
    max_alpha: float = 0.25
    min_alpha: float = 0.0
    state_colors: tuple[str, ...] = ("#4C78A8", "#F58518", "#54A24B", "#E45756")

def _load_oos_regime_probs(artifact_dir: Path) -> pd.DataFrame:
    fp = artifact_dir / "oos_regime_probs.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, parse_dates=["date"]).set_index("date")
    return df

def _load_price_series(data_fp: Path, cfg: RegimeShadingconfig) -> pd.Series:
    df = pd.read_csv(data_fp, parse_dates = [cfg.date_col])
    df = df.sort_values(cfg.date_col).set_index(cfg.date_col)
    return df[cfg.price_col].astype(float)

def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()

def _shade(ax, x, probs, cfg: RegimeShadingconfig):
    state = probs.argmax(axis=1)

    if cfg.use_soft:
        alpha = probs.max(axis=1)
        alpha = np.clip(alpha, cfg.min_alpha, cfg.max_alpha)
    else:
        alpha = np.full(len(x), cfg.max_alpha)
    
    start = 0
    for t in range(1, len(x)+1):
        if t == len(x) or state[t] != state[start]:
            k = int(state[start])
            ax.axvspan(
                x[start],
                x[t-1],
                color=cfg.state_colors[k % len(cfg.state_colors)],
                alpha=float(alpha[start])
                linewidth=0
            )
            start = t

def make_regime_shading_plot(cfg: RegimeShadingconfig,
                             data_fp: Path,
                             title: str | None = None,)-> Path: 
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    probs_df = _load_oos_regime_probs(cfg.artifact_dir)
    price = _load_price_series(data_fp, cfg)

    df = pd.DataFrame({"price": price}).join(probs_df, how="inner").dropna()

    prob_cols = [c for c in df.columns if c.startswith("p_state_")]
    probs = df[prob_cols].to_numpy()

    returns = df["price"].pct_change()
    rv = _realized_vol(returns, cfg.rv_window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = df.index

    _shade(ax1, x, probs, cfg)
    ax1.plot(x, np.log(df["price"]), lw=1.2)
    ax1.set_ylabel("log price")

    _shade(ax2, x, probs, cfg)
    ax2.plot(x, rv, lw=1.2)
    ax2.set_ylabel(f"realized vol ({cfg.rv_window}d)")
    ax2.set_xlabel("date")

    if title:
        fig.suptitle(title)
    
    fig.tight_layout()
    out_fp = cfg.out_dir / "regime_shading.png"
    fig.savefig(out_fp, dpi=200)
    plt.close(fig)
    return out_fp