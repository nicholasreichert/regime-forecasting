"""Model Confidence Set and economic evaluation over the cross-asset predictions.

Both operate on the stored prediction series, so neither refits anything.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.features import build_feature_table
from src.data.make_dataset import load_or_download_equity_data
from src.eval.economic import (
    forecast_to_daily_vol,
    next_period_returns,
    var_backtest,
    volatility_targeting,
)
from src.eval.mcs import mcs_over_models

PRED_ROOT = Path("artifacts") / "study" / "multi_asset" / "predictions"
OUT = Path("artifacts") / "study" / "multi_asset"
HORIZONS = [1, 5, 20]

# Models entered into the MCS. The trivial baselines are excluded: they would be
# eliminated in the first round and only inflate the bootstrap range statistic.
MCS_MODELS = ["ridge", "har", "garch11", "ewma", "regime_normal", "regime_features",
              "regime_shrunk", "gbm_pooled", "gbm_gated"]


def _load_preds(ticker: str, h: int) -> Dict[str, pd.DataFrame]:
    d = PRED_ROOT / ticker / f"h{h}"
    if not d.exists():
        return {}
    return {fp.stem: pd.read_csv(fp, index_col=0, parse_dates=True) for fp in d.glob("*.csv")}


def run_mcs(alpha: float = 0.10, n_boot: int = 2000) -> pd.DataFrame:
    rows: List[dict] = []
    tickers = sorted(p.name for p in PRED_ROOT.iterdir() if p.is_dir())

    for ticker in tickers:
        for h in HORIZONS:
            preds = _load_preds(ticker, h)
            usable = {k: v for k, v in preds.items() if k in MCS_MODELS}
            if len(usable) < 3:
                continue

            idx = None
            for v in usable.values():
                idx = v.index if idx is None else idx.intersection(v.index)
            y = usable[next(iter(usable))].loc[idx, "y_true"].to_numpy()
            P = {k: v.loc[idx, "y_pred"].to_numpy() for k, v in usable.items()}

            for loss in ("mse", "qlike"):
                try:
                    res = mcs_over_models(y, P, loss=loss, alpha=alpha,
                                          horizon=h, n_boot=n_boot, seed=0)
                except Exception as e:
                    print(f"[warn] MCS {ticker} h={h} {loss}: {type(e).__name__}: {e}")
                    continue
                for m in usable:
                    rows.append({
                        "ticker": ticker, "horizon": h, "loss": loss, "model": m,
                        "in_mcs": m in res.included,
                        "mcs_p": res.p_values.get(m, np.nan),
                        "set_size": len(res.included),
                    })
            print(f"[mcs] {ticker:<8} h={h:2d} kept {len(res.included)}/{len(usable)}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "mcs_results.csv", index=False)
    print(f"\nWrote {OUT / 'mcs_results.csv'}")
    return df


def run_economic(target_ann_vol: float = 0.10, var_alpha: float = 0.05) -> pd.DataFrame:
    """Volatility targeting at every horizon; VaR at h=1 where returns are signed.

    The VaR exercise needs the realised signed return over the period being
    covered, which is unambiguous only at h=1; at longer horizons the forecast
    describes average volatility over a window rather than a single period.
    """
    cfg = load_config()
    rows: List[dict] = []
    tickers = sorted(p.name for p in PRED_ROOT.iterdir() if p.is_dir())

    for ticker in tickers:
        c = replace(cfg, data=replace(cfg.data, ticker=ticker))
        try:
            data = build_feature_table(c, load_or_download_equity_data(c).df)
        except Exception as e:
            print(f"[skip] {ticker}: {type(e).__name__}: {e}")
            continue
        nxt = next_period_returns(data)

        for h in HORIZONS:
            for model, dfp in _load_preds(ticker, h).items():
                sigma = forecast_to_daily_vol(dfp["y_pred"], h)
                vt = volatility_targeting(sigma, nxt, target_ann_vol=target_ann_vol)
                row = {
                    "ticker": ticker, "horizon": h, "model": model,
                    "realized_ann_vol": vt.realized_ann_vol,
                    "vol_error": vt.vol_error,
                    "sharpe": vt.sharpe,
                    "max_drawdown": vt.max_drawdown,
                    "mean_leverage": vt.mean_leverage,
                    "turnover": vt.turnover,
                }
                if h == 1:
                    vr = var_backtest(sigma, nxt, alpha=var_alpha)
                    row.update({
                        "var_violation_rate": vr.violation_rate,
                        "var_p_uc": vr.p_uc,
                        "var_p_ind": vr.p_ind,
                        "var_p_cc": vr.p_cc,
                        "var_mean_level": vr.mean_var,
                    })
                rows.append(row)
        print(f"[econ] {ticker}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "economic_results.csv", index=False)
    print(f"\nWrote {OUT / 'economic_results.csv'}")
    return df


def summarise(mcs: pd.DataFrame, econ: pd.DataFrame) -> None:
    print("\n=== Model Confidence Set: share of asset-horizon pairs retained ===")
    for loss in ("mse", "qlike"):
        s = mcs[mcs["loss"] == loss]
        if s.empty:
            continue
        print(f"  -- {loss} --")
        g = s.groupby("model")["in_mcs"].agg(["mean", "sum", "count"])
        for m in g.sort_values("mean", ascending=False).index:
            r = g.loc[m]
            print(f"    {m:<18} {100 * r['mean']:5.1f}%   ({int(r['sum'])}/{int(r['count'])})")

    print("\n=== Volatility targeting: |realized - target| annualised vol ===")
    for h in HORIZONS:
        s = econ[econ["horizon"] == h]
        if s.empty:
            continue
        g = s.groupby("model")["vol_error"].median().sort_values()
        print(f"  -- h={h} (lower is better) --")
        for m, v in g.items():
            print(f"    {m:<18} {v:.4f}")

    e1 = econ[econ["horizon"] == 1]
    if "var_p_cc" in e1.columns and e1["var_p_cc"].notna().any():
        print("\n=== VaR (5%) at h=1 ===")
        g = e1.groupby("model").agg(
            rate=("var_violation_rate", "median"),
            pass_cc=("var_p_cc", lambda x: float((x > 0.05).mean())),
            pass_ind=("var_p_ind", lambda x: float((x > 0.05).mean())),
            level=("var_mean_level", "median"),
        )
        for m in g.index:
            r = g.loc[m]
            print(f"    {m:<18} rate={r['rate']:.4f}  pass CC={100 * r['pass_cc']:5.1f}%  "
                  f"pass IND={100 * r['pass_ind']:5.1f}%  mean VaR={r['level']:.5f}")


TABLES = Path("paper") / "tables"

DISPLAY = {
    "ridge": "Ridge (pooled)",
    "har": "HAR",
    "garch11": "GARCH(1,1)",
    "ewma": "EWMA",
    "gbm_pooled": "Gradient boosting (pooled)",
    "regime_normal": r"\textbf{Regime gating}",
    "regime_features": r"\textbf{Regime as features}",
    "regime_shrunk": r"\textbf{Regime gating, shrunk}",
    "gbm_gated": r"\textbf{Regime gating, GBM experts}",
}
ROW_ORDER = ["ridge", "har", "garch11", "ewma", "gbm_pooled",
             "regime_normal", "regime_features", "regime_shrunk", "gbm_gated"]


def make_mcs_table(mcs: pd.DataFrame) -> None:
    """Share of asset-horizon pairs in which each model survives into the MCS."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Model Confidence Set \cite{hansen2011mcs} at the 90\% level,"
        r" computed separately for each of the 60 asset-horizon pairs with a"
        r" stationary block bootstrap at block length $h$. Entries are the"
        r" percentage of pairs in which the model survives into the set of models"
        r" that cannot be distinguished from the best. Regime-conditioned models"
        r" are in bold. Unlike the pairwise tests, this accounts for the full set"
        r" of competitors simultaneously.}",
        r"\label{tab:mcs}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Model & MSE & QLIKE \\",
        r"\midrule",
    ]
    for m in ROW_ORDER:
        cells = []
        for loss in ("mse", "qlike"):
            s = mcs[(mcs["model"] == m) & (mcs["loss"] == loss)]
            cells.append(f"{100 * s['in_mcs'].mean():.0f}\\%" if len(s) else "--")
        if m == "regime_normal":
            lines.append(r"\midrule")
        lines.append(f"{DISPLAY.get(m, m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "mcs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'mcs.tex'}")


def make_economic_table(econ: pd.DataFrame) -> None:
    """Decision-relevant performance: volatility targeting and VaR."""
    e1 = econ[econ["horizon"] == 1]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Economic evaluation, median over 20 assets. Volatility"
        r" targeting sizes a position at $\sigma^{\text{target}}/\hat\sigma_t$ for"
        r" a 10\% annualised target; the reported error is"
        r" $|\text{realised}-\text{target}|$ annualised volatility, so lower is"
        r" better. VaR is at the 5\% level with the forecast-to-quantile scaling"
        r" calibrated by filtered historical simulation on a trailing window,"
        r" evaluated at $h{=}1$ where the realised signed return is unambiguous."
        r" ``Pass CC'' is the share of assets passing Christoffersen conditional"
        r" coverage at 5\%.}",
        r"\label{tab:economic}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Vol-target error} & \multicolumn{2}{c}{VaR (5\%, $h{=}1$)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"Model & $h{=}1$ & $h{=}5$ & $h{=}20$ & Rate & Pass CC \\",
        r"\midrule",
    ]
    for m in ROW_ORDER:
        cells = []
        for h in HORIZONS:
            s = econ[(econ["model"] == m) & (econ["horizon"] == h)]["vol_error"]
            cells.append(f"{s.median():.4f}" if len(s) else "--")
        s1 = e1[e1["model"] == m]
        if len(s1) and "var_violation_rate" in s1.columns and s1["var_violation_rate"].notna().any():
            cells.append(f"{s1['var_violation_rate'].median():.3f}")
            cells.append(f"{100 * (s1['var_p_cc'] > 0.05).mean():.0f}\\%")
        else:
            cells += ["--", "--"]
        if m == "regime_normal":
            lines.append(r"\midrule")
        lines.append(f"{DISPLAY.get(m, m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "economic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'economic.tex'}")


def main() -> None:
    mcs = run_mcs()
    econ = run_economic()
    summarise(mcs, econ)
    make_mcs_table(mcs)
    make_economic_table(econ)


if __name__ == "__main__":
    main()
