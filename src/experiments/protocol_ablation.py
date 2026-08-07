"""How much of the apparent regime-switching gain is manufactured by the protocol?

Four methodological choices, each individually defensible-looking, are common in
applied regime-switching work:

P0  the multi-horizon target is a point-in-time transform of the single return
    h days ahead rather than realized volatility over [t+1, t+h];
P1  the linear baseline is fitted on unstandardized features at a fixed penalty,
    so it is weaker than it needs to be;
P2  no embargo separates train from test, so training rows whose targets reach
    into the test fold leak;
P3  the number of regimes and the gating rule are chosen by out-of-sample score,
    and that same score is then reported.

This module starts from all four in place and removes them one at a time,
reporting the regime model's improvement over its matched pooled-ridge baseline
at each stage. The point is not that any single choice is fatal, but that they
compound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.dm_test import diebold_mariano
from src.eval.metrics import rmse
from src.eval.oos import (
    ABLATION_BY_NAME,
    EvalSpec,
    HMMCache,
    _apply_prob_transform,
    _splits,
)
from src.eval.walk_forward import inner_validation_split
from src.models.regime_conditioned import RegimeConditionedRidge


@dataclass(frozen=True)
class Protocol:
    name: str
    label: str
    target_kind: str  # "absret" (aliased) or "rv"
    standardize: bool  # standardized + GCV-tuned ridge everywhere
    embargo: bool  # purge train rows whose targets overlap the test fold
    nested_selection: bool  # choose K/mode on inner validation instead of on test


PROTOCOLS: tuple[Protocol, ...] = (
    Protocol("P0", "as-published", "absret", False, False, False),
    Protocol("P1", "+ tuned/standardized baseline", "absret", True, False, False),
    Protocol("P2", "+ train/test embargo", "absret", True, True, False),
    Protocol("P3", "+ nested selection", "absret", True, True, True),
    Protocol("P4", "+ realized-vol target", "rv", True, True, True),
)


def _fit_predict_fold(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    train_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    K: int,
    mode: str,
    cache: HMMCache,
    standardize: bool,
    seed: int,
    split_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the regime model on one fold and return (y_true, y_pred) for the test block."""
    train = df.loc[train_idx]
    test = df.loc[test_idx]

    hres = cache.get(train, test, K, seed)
    n_tr = min(len(train), hres.train_probs.shape[0])
    n_te = min(len(test), hres.test_probs.shape[0])

    Xtr = train[list(features)].iloc[-n_tr:]
    ytr = train[target].iloc[-n_tr:]
    te = test.iloc[-n_te:]

    rng_tr = np.random.default_rng(seed + 10_000 + split_idx)
    rng_te = np.random.default_rng(seed + 20_000 + split_idx)
    ab = ABLATION_BY_NAME["normal"]
    ptr = _apply_prob_transform(hres.train_probs[-n_tr:], ab.train, rng_tr)
    pte = _apply_prob_transform(hres.test_probs[-n_te:], ab.test, rng_te)

    alpha = None if standardize else 1.0
    m = RegimeConditionedRidge(
        alpha=alpha, mode=mode, min_points_per_regime=200, standardize=standardize
    )
    m.fit(Xtr, ytr, ptr)
    pred = m.predict(te[list(features)], pte)
    return te[target].to_numpy(dtype=float), np.asarray(pred, dtype=float)


def _pooled_ridge_oos(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    ev: EvalSpec,
    standardize: bool,
) -> tuple[pd.Series, pd.Series]:
    """Matched pooled-ridge baseline: same splits, same embargo, same scaling policy."""
    from sklearn.linear_model import Ridge

    from src.models.baselines import RidgeBaseline

    yt_parts, yp_parts = [], []
    for split in _splits(df, ev):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]
        if standardize:
            model = RidgeBaseline(alpha=None)
            model.fit(train[list(features)], train[target])
            pred = model.predict(test[list(features)])
        else:
            reg = Ridge(alpha=1.0)
            reg.fit(train[list(features)], train[target])
            pred = reg.predict(test[list(features)])
        yt_parts.append(test[target].astype(float))
        yp_parts.append(pd.Series(np.asarray(pred, dtype=float), index=test.index))
    return pd.concat(yt_parts).sort_index(), pd.concat(yp_parts).sort_index()


def run_protocol(
    df: pd.DataFrame,
    features: Sequence[str],
    h: int,
    proto: Protocol,
    cache: HMMCache,
    K_values: Sequence[int],
    modes: Sequence[str],
    target_type: str,
    seed: int,
) -> dict:
    target = f"y_rv_h{h}" if proto.target_kind == "rv" else f"y_{target_type}_h{h}"
    ev = EvalSpec(6, 1, 1, embargo=h if proto.embargo else 0)

    splits = _splits(df, ev)

    if proto.nested_selection:
        # choose K and mode per fold using an inner validation block
        yt_parts, yp_parts, chosen = [], [], []
        for i, split in enumerate(splits):
            inner_tr, inner_val = inner_validation_split(
                split.train_idx, val_fraction=0.25, embargo=ev.embargo
            )
            best, best_score = None, np.inf
            for K in K_values:
                for mode in modes:
                    yv, pv = _fit_predict_fold(
                        df, features, target, inner_tr, inner_val, K, mode,
                        cache, proto.standardize, seed, i,
                    )
                    s = rmse(yv, pv)
                    if np.isfinite(s) and s < best_score:
                        best_score, best = s, (K, mode)
            if best is None:
                best = (int(min(K_values)), modes[0])
            chosen.append(best)
            yt, yp = _fit_predict_fold(
                df, features, target, split.train_idx, split.test_idx,
                best[0], best[1], cache, proto.standardize, seed, i,
            )
            yt_parts.append(pd.Series(yt, index=split.test_idx))
            yp_parts.append(pd.Series(yp, index=split.test_idx))
        y_true = pd.concat(yt_parts).sort_index()
        y_pred = pd.concat(yp_parts).sort_index()
        sel_note = f"nested (modal K={max(set(k for k,_ in chosen), key=[k for k,_ in chosen].count)})"
    else:
        # the biased protocol: run the whole grid out of sample, keep the winner,
        # and report the winner's own out-of-sample score
        best, best_score, best_series = None, np.inf, None
        for K in K_values:
            for mode in modes:
                yt_parts, yp_parts = [], []
                for i, split in enumerate(splits):
                    yt, yp = _fit_predict_fold(
                        df, features, target, split.train_idx, split.test_idx,
                        K, mode, cache, proto.standardize, seed, i,
                    )
                    yt_parts.append(pd.Series(yt, index=split.test_idx))
                    yp_parts.append(pd.Series(yp, index=split.test_idx))
                yt_all = pd.concat(yt_parts).sort_index()
                yp_all = pd.concat(yp_parts).sort_index()
                s = rmse(yt_all.to_numpy(), yp_all.to_numpy())
                if np.isfinite(s) and s < best_score:
                    best_score, best, best_series = s, (K, mode), (yt_all, yp_all)
        y_true, y_pred = best_series
        sel_note = f"best-on-test (K={best[0]}, {best[1]})"

    yb_true, yb_pred = _pooled_ridge_oos(df, features, target, ev, proto.standardize)
    common = y_true.index.intersection(yb_true.index)

    r_regime = rmse(y_true.loc[common].to_numpy(), y_pred.loc[common].to_numpy())
    r_base = rmse(yb_true.loc[common].to_numpy(), yb_pred.loc[common].to_numpy())

    dm = diebold_mariano(
        y_true.loc[common].to_numpy(),
        y_pred.loc[common].to_numpy(),
        yb_pred.loc[common].to_numpy(),
        horizon=h,
        loss="mse",
    )

    return {
        "protocol": proto.name,
        "protocol_label": proto.label,
        "horizon": h,
        "target": target,
        "selection": sel_note,
        "rmse_regime": r_regime,
        "rmse_pooled_ridge": r_base,
        # positive => the regime model looks better than its matched baseline
        "improvement_pct": 100.0 * (r_base - r_regime) / r_base,
        "dm_stat": dm.stat,
        "dm_p": dm.p_value,
        "n_oos": int(len(common)),
    }


def main(out_dir: Path | None = None) -> Path:
    cfg = load_config()
    df = build_and_save_processed_dataset(cfg).df
    features = [c for c in df.columns if not c.startswith("y_")]

    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type,
        n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol,
        min_covar=cfg.hmm.min_covar,
    )
    cache = HMMCache(hmm_kwargs)

    rows: List[dict] = []
    for h in [int(x) for x in cfg.targets.horizons]:
        for proto in PROTOCOLS:
            r = run_protocol(
                df, features, h, proto, cache,
                K_values=[int(k) for k in cfg.hmm.K_values],
                modes=["hard", "soft"],
                target_type=cfg.targets.target_type,
                seed=cfg.project.seed,
            )
            rows.append(r)
            print(
                f"h={h} {r['protocol']:<3} {r['protocol_label']:<30} "
                f"regime={r['rmse_regime']:.6f} base={r['rmse_pooled_ridge']:.6f} "
                f"improvement={r['improvement_pct']:+.2f}%  p={r['dm_p']:.3f}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out_dir = out_dir or (Path("artifacts") / "study" / "protocol_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "protocol_ablation.csv", index=False)
    print(f"\nWrote {out_dir / 'protocol_ablation.csv'}")
    print(f"[hmm cache] {cache.hits} hits / {cache.misses} fits")
    return out_dir


if __name__ == "__main__":
    main()
