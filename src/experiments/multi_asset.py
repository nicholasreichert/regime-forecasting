"""Does the null hold across assets, or only for SPY?

The single-asset result is the weakest part of the study: SPY is the most liquid
and most-studied series there is, and the one where trailing realized volatility
is most informative, so it is also the setting where regime labels have least
room to add anything. This module reruns the corrected protocol across a
cross-section of assets chosen to span the dimensions that plausibly matter -
liquidity, volatility level, asset class, and how well-behaved the volatility
dynamics are.

Two outcomes are both informative. If the null holds everywhere, "you only tested
SPY" stops being an objection. If regime conditioning pays somewhere, the paper
becomes a characterisation of *when* it pays, which is a better paper than a
flat null.

Assets are run in parallel; each worker is pinned to a single BLAS thread to
avoid oversubscription.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import List

import pandas as pd

# Must be set before numpy/sklearn import in worker processes.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.features import build_feature_table  # noqa: E402
from src.data.make_dataset import load_or_download_equity_data  # noqa: E402
from src.eval.dm_test import diebold_mariano, equivalence_bound  # noqa: E402
from src.eval.oos import (  # noqa: E402
    ABLATION_BY_NAME,
    EvalSpec,
    HMMCache,
    collect_oos_baseline,
    collect_oos_regime_nested,
    collect_oos_regime_variant,
    compute_metrics,
)
from src.models.baselines import (  # noqa: E402
    EWMAVolBaseline,
    GARCHBaseline,
    HARBaseline,
    RandomWalkVolBaseline,
    RidgeBaseline,
)

# Chosen to span asset class, volatility level, and liquidity. Start dates vary;
# assets with too little history for a walk-forward are skipped automatically.
ASSETS: dict[str, str] = {
    # broad equity
    "SPY": "US large cap",
    "QQQ": "US tech-heavy",
    "IWM": "US small cap",
    "EFA": "Developed ex-US",
    "EEM": "Emerging markets",
    # sectors
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    # single names
    "AAPL": "Single name (tech)",
    "MSFT": "Single name (tech)",
    "JPM": "Single name (financial)",
    "XOM": "Single name (energy)",
    "KO": "Single name (staples)",
    "TSLA": "Single name (high vol)",
    # rates and credit
    "TLT": "Long treasuries",
    "HYG": "High yield credit",
    # commodities
    "GLD": "Gold",
    "USO": "Crude oil",
    # currency and crypto
    "FXE": "EUR/USD",
    "BTC-USD": "Bitcoin",
}

HORIZONS = [1, 5, 20]
RW_COL = {1: "rv_bwd_1", 5: "rv_bwd_5", 20: "rv_bwd_22"}

# Trimmed relative to the SPY study: the trivial baselines (zero, train mean) and
# the two ablation arms that are not needed for the cross-sectional claim are
# dropped, since 20 assets x 3 horizons is otherwise a lot of compute for rows
# nobody reads.
REGIME_ARMS = ["normal", "gate_shuffle", "single"]


def run_asset(ticker: str) -> List[dict]:
    cfg = load_config()
    cfg = replace(cfg, data=replace(cfg.data, ticker=ticker))

    try:
        raw = load_or_download_equity_data(cfg)
        data = build_feature_table(cfg, raw.df)
    except Exception as e:
        return [{"ticker": ticker, "error": f"data: {type(e).__name__}: {e}"}]

    features = [c for c in data.columns if not c.startswith("y_")]
    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type,
        n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol,
        min_covar=cfg.hmm.min_covar,
    )
    cache = HMMCache(hmm_kwargs)

    rows: List[dict] = []
    for h in HORIZONS:
        target = f"y_rv_h{h}"
        ev = EvalSpec(
            train_years=cfg.evaluation.train_years,
            test_years=cfg.evaluation.test_years,
            step_years=cfg.evaluation.step_years,
            embargo=h,
        )

        baselines = {
            "rw_vol": lambda: RandomWalkVolBaseline(RW_COL[h]),
            "ridge": lambda: RidgeBaseline(alpha=None),
            "har": HARBaseline,
            "ewma": lambda: EWMAVolBaseline(horizon=h),
            "garch11": lambda: GARCHBaseline(horizon=h),
        }

        preds = {}
        for name, factory in baselines.items():
            try:
                preds[name] = collect_oos_baseline(data, features, target, factory, ev)
            except Exception as e:
                rows.append({"ticker": ticker, "horizon": h, "model": name,
                             "error": f"{type(e).__name__}: {e}"})

        for arm in REGIME_ARMS:
            try:
                preds[f"regime_{arm}"] = collect_oos_regime_nested(
                    df=data, features=features, target=target, ev=ev,
                    K_values=[int(k) for k in cfg.hmm.K_values],
                    modes=["hard", "soft"],
                    ablation=ABLATION_BY_NAME[arm],
                    cache=cache, seed=cfg.project.seed,
                )
            except Exception as e:
                rows.append({"ticker": ticker, "horizon": h, "model": f"regime_{arm}",
                             "error": f"{type(e).__name__}: {e}"})

        # The two ways of using the regime signal that avoid gating entirely.
        for variant in ("features", "shrunk"):
            try:
                preds[f"regime_{variant}"] = collect_oos_regime_variant(
                    df=data, features=features, target=target, ev=ev,
                    K_values=[int(k) for k in cfg.hmm.K_values],
                    variant=variant, cache=cache, seed=cfg.project.seed,
                )
            except Exception as e:
                rows.append({"ticker": ticker, "horizon": h, "model": f"regime_{variant}",
                             "error": f"{type(e).__name__}: {e}"})

        if "regime_single" not in preds or "regime_normal" not in preds:
            continue

        ref = preds["regime_single"]  # the pooled-ridge control
        ref_rmse = float(np.sqrt(np.mean((ref.y_true - ref.y_pred) ** 2)))

        for name, res in preds.items():
            m = compute_metrics(res, data)
            row = {
                "ticker": ticker,
                "asset_class": ASSETS.get(ticker, ""),
                "horizon": h,
                "model": name,
                "n_oos": m["n_oos"],
                "rmse": m["rmse"],
                "qlike": m["qlike"],
                "mean_vol": float(data["ret_vol_20"].mean()),
                "sample_start": str(data.index.min().date()),
                "sample_end": str(data.index.max().date()),
            }
            # improvement over the pooled-ridge control; positive => better
            row["improvement_pct"] = 100.0 * (ref_rmse - m["rmse"]) / ref_rmse

            if name != "regime_single":
                common = res.y_true.index.intersection(ref.y_true.index)
                yt = res.y_true.loc[common].to_numpy()
                pa = res.y_pred.loc[common].to_numpy()
                pb = ref.y_pred.loc[common].to_numpy()
                dm = diebold_mariano(yt, pa, pb, horizon=h, loss="mse")
                row["dm_stat_vs_pooled"] = dm.stat
                row["dm_p_vs_pooled"] = dm.p_value
                # how large an improvement the data can actually exclude
                eb = equivalence_bound(yt, pa, pb, horizon=h)
                row["ci_lower_pct"] = eb.lower
                row["ci_upper_pct"] = eb.upper
            for k, v in (res.diagnostics or {}).items():
                row[k] = v
            rows.append(row)

    return rows


def prefetch(tickers: List[str]) -> List[str]:
    """Download and cache raw data serially before fanning out.

    Twenty workers hitting Yahoo at once invites rate limiting, and a throttled
    download failure is indistinguishable from an asset with no history. Fetching
    up front also lets us drop assets with too little data before spending
    compute on them.
    """
    cfg = load_config()
    usable = []
    for t in tickers:
        c = replace(cfg, data=replace(cfg.data, ticker=t))
        try:
            raw = load_or_download_equity_data(c)
            feats = build_feature_table(c, raw.df)
            years = (feats.index.max() - feats.index.min()).days / 365.25
            need = cfg.evaluation.train_years + cfg.evaluation.test_years
            if years < need + 0.5:
                print(f"[skip] {t:<8} only {years:.1f}y of history (need >{need})", flush=True)
                continue
            usable.append(t)
            print(f"[data] {t:<8} {len(feats):5d} rows  {feats.index.min().date()} -> "
                  f"{feats.index.max().date()}  ({years:.1f}y)", flush=True)
        except Exception as e:
            print(f"[skip] {t:<8} {type(e).__name__}: {e}", flush=True)
    return usable


def main(out_dir: Path | None = None, workers: int = 10) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = out_dir or (Path("artifacts") / "study" / "multi_asset")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== prefetch ===", flush=True)
    tickers = prefetch(list(ASSETS))
    print(f"\n=== running {len(tickers)} assets on {workers} workers ===", flush=True)
    all_rows: List[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_asset, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[fail] {t}: {type(e).__name__}: {e}", flush=True)
                continue
            all_rows.extend(rows)

            ok = [r for r in rows if "error" not in r and r.get("model") == "regime_normal"]
            if ok:
                msg = "  ".join(f"h={r['horizon']}:{r['improvement_pct']:+.2f}%" for r in ok)
                print(f"[done] {t:<8} n={ok[0]['n_oos']:5d}  {msg}", flush=True)
            else:
                errs = {r.get("error") for r in rows if "error" in r}
                print(f"[skip] {t:<8} {errs}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "multi_asset_results.csv", index=False)
    print(f"\nWrote {out_dir / 'multi_asset_results.csv'}  ({len(df)} rows)")
    return out_dir


if __name__ == "__main__":
    main()
