"""What does a Gaussian HMM actually learn from financial returns?

The applied result of this paper is that regime conditioning does not improve
volatility forecasts. On its own that is uninterpretable: it could mean markets
have no regimes, or that our pipeline cannot find them. This module settles the
question by running the identical pipeline on synthetic data from worlds where
we know the answer.

Five data-generating processes, all calibrated to roughly SPY's unconditional
volatility and sample length:

``ms_slope``    True Markov switching where the states differ in the *predictive
                relationship* - the mean-reversion speed of volatility - not just
                its level. This is precisely what a mixture of per-regime experts
                is designed to exploit, and is our positive control.
``ms_level``    True Markov switching where states differ only in volatility
                level. Regimes are real but a continuous volatility feature
                already summarises them.
``garch11``     Persistent continuous volatility, no discrete states anywhere.
``iid_t``       Fat tails, no dynamics of any kind.
``iid_normal``  Nothing to find. Null control.

The diagnostics are the ones the applied literature reports - separated state
means, persistent transitions, states that line up with volatility clusters -
plus the one it usually does not: whether conditioning on the inferred state
actually improves a forecast.

The question the study answers is not "does the HMM find regimes" (it will, in
every world including the ones with none) but "do the standard diagnostics
distinguish a world with regimes from a world without".
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.features import build_feature_table  # noqa: E402
from src.eval.metrics import rmse  # noqa: E402
from src.eval.oos import (  # noqa: E402
    ABLATION_BY_NAME,
    EvalSpec,
    HMMCache,
    collect_oos_regime_nested,
)
from src.regime.hmm import fit_hmm_and_infer_probs  # noqa: E402

N_OBS = 5391  # matches the SPY sample
TARGET_DAILY_VOL = 0.011
HORIZONS = [1, 5, 20]
K_DIAG = 3  # K used for the descriptive diagnostics


# --------------------------------------------------------------------------
# data-generating processes
# --------------------------------------------------------------------------


def dgp_iid_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    _TRUE_STATE.pop("last", None)
    return rng.normal(scale=TARGET_DAILY_VOL, size=n)


def dgp_iid_t(n: int, rng: np.random.Generator, df: int = 4) -> np.ndarray:
    """Fat tails, zero dynamics. A Gaussian mixture can fit this marginal well
    using several components, so any 'regimes' found here are pure artefact."""
    _TRUE_STATE.pop("last", None)
    x = rng.standard_t(df, size=n)
    return x / np.sqrt(df / (df - 2)) * TARGET_DAILY_VOL


def dgp_garch11(n: int, rng: np.random.Generator,
                alpha: float = 0.11, beta: float = 0.88, df: int = 6) -> np.ndarray:
    """Persistent continuous volatility. No discrete state exists at any point.

    Student-t innovations, because this world has to be a *fair* stand-in for
    real returns: with Gaussian innovations it reproduces the volatility
    clustering but not the fat tails, and the argument would then be open to the
    objection that the HMM is distinguishing the two on kurtosis rather than on
    dynamics. Calibrated to roughly SPY's unconditional volatility, excess
    kurtosis and absolute-return autocorrelation.
    """
    _TRUE_STATE.pop("last", None)
    omega = TARGET_DAILY_VOL**2 * (1.0 - alpha - beta)
    scale = np.sqrt(df / (df - 2))
    r = np.empty(n)
    s2 = TARGET_DAILY_VOL**2
    for t in range(n):
        e = rng.standard_t(df) / scale
        r[t] = np.sqrt(s2) * e
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def _markov_path(n: int, p_stay: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = rng.random() < p_stay[s[t - 1]]
        s[t] = s[t - 1] if stay else 1 - s[t - 1]
    return s


# DGPs record the latent state they generated (None when there isn't one), so the
# study can ask whether the HMM recovers a regime that genuinely exists.
_TRUE_STATE: Dict[str, np.ndarray] = {}


def dgp_ms_level(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genuine two-state switching; states differ only in volatility level."""
    s = _markov_path(n, (0.99, 0.97), rng)
    _TRUE_STATE["last"] = s
    sig = np.where(s == 0, 0.006, 0.022)
    return rng.normal(scale=sig)


def dgp_ms_slope(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genuine switching where the states differ in volatility *dynamics*.

    Within each state, log-volatility is AR(1) with a state-specific
    mean-reversion speed and long-run level. A per-regime linear expert has
    genuinely different coefficients to learn in each state, which is the
    situation a regime-conditioned mixture exists for. This is the positive
    control: if regime conditioning does not help here, the architecture is at
    fault rather than the data.
    """
    s = _markov_path(n, (0.99, 0.97), rng)
    _TRUE_STATE["last"] = s
    phi = np.array([0.95, 0.70])       # persistence of log-vol per state
    mu = np.log(np.array([0.007, 0.022]))  # long-run log-vol per state
    log_s = np.empty(n)
    log_s[0] = mu[s[0]]
    for t in range(1, n):
        k = s[t]
        log_s[t] = (1 - phi[k]) * mu[k] + phi[k] * log_s[t - 1] + rng.normal(scale=0.12)
    return rng.normal(scale=np.exp(log_s))


DGPS: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "ms_slope": dgp_ms_slope,
    "ms_level": dgp_ms_level,
    "garch11": dgp_garch11,
    "iid_t": dgp_iid_t,
    "iid_normal": dgp_iid_normal,
}

DGP_HAS_REGIMES = {
    "ms_slope": True, "ms_level": True,
    "garch11": False, "iid_t": False, "iid_normal": False,
}


# --------------------------------------------------------------------------


def _frame_from_returns(returns: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Wrap a return series as OHLCV so the production feature builder can be reused.

    Using the real pipeline rather than a bespoke one guarantees the synthetic
    and empirical arms see identical feature definitions.
    """
    n = len(returns)
    idx = pd.bdate_range("2005-01-03", periods=n)
    price = 100.0 * np.exp(np.cumsum(returns))
    volume = np.exp(rng.normal(loc=16.0, scale=0.3, size=n))  # non-degenerate
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price,
         "Close": price, "Adj Close": price, "Volume": volume},
        index=idx,
    )


def hmm_diagnostics(data: pd.DataFrame, cfg, seed: int,
                    true_state: np.ndarray | None = None) -> dict:
    """The descriptive checks an applied paper typically reports, plus two it does not.

    The two additions are (a) whether the inferred state actually corresponds to
    the latent state when one exists, and (b) how much of the apparent transition
    persistence survives dropping the overlapping rolling-volatility emission.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(data)
    train = data.iloc[: int(0.7 * n)]
    test = data.iloc[int(0.7 * n):]

    res = fit_hmm_and_infer_probs(
        train_df=train, test_df=test, K=K_DIAG,
        covariance_type=cfg.hmm.covariance_type, n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol, min_covar=cfg.hmm.min_covar, seed=seed,
    )
    A = res.model.transmat_
    evals, evecs = np.linalg.eig(A.T)
    stat = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
    stat = stat / stat.sum()

    probs = res.test_probs
    states = probs.argmax(axis=1)
    te = test.iloc[-len(states):]
    vol = te["ret_vol_20"].to_numpy(dtype=float)

    mu = np.array([vol[states == k].mean() if (states == k).any() else np.nan
                   for k in range(K_DIAG)])
    finite = np.isfinite(mu)

    # Can trailing realized volatility alone reproduce the HMM's state calls?
    rv_cols = ["rv_bwd_1", "rv_bwd_5", "rv_bwd_22"]
    if len(set(states)) > 1:
        clf = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
        clf.fit(te[rv_cols], states)
        redundancy = float(clf.score(te[rv_cols], states))
        base_rate = float(np.bincount(states).max() / len(states))
    else:
        redundancy, base_rate = np.nan, 1.0

    # If states were temporally independent, P(z_t=k | z_{t-1}) would equal the
    # stationary probability. Excess self-transition is what "persistence" means.
    excess = float(np.mean(np.diag(A) - stat))

    # How much persistence survives removing the overlapping-window emission?
    res_nv = fit_hmm_and_infer_probs(
        train_df=train, test_df=test, K=K_DIAG,
        covariance_type=cfg.hmm.covariance_type, n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol, min_covar=cfg.hmm.min_covar, seed=seed, include_vol=False,
    )
    A_nv = res_nv.model.transmat_
    ev_nv, evec_nv = np.linalg.eig(A_nv.T)
    stat_nv = np.real(evec_nv[:, np.argmin(np.abs(ev_nv - 1))])
    stat_nv = stat_nv / stat_nv.sum()
    excess_nv = float(np.mean(np.diag(A_nv) - stat_nv))

    # Does the inferred state track the latent one, where a latent one exists?
    if true_state is not None:
        from sklearn.metrics import adjusted_rand_score
        ts = np.asarray(true_state)[-len(te):]
        state_recovery = float(adjusted_rand_score(ts[-len(states):], states))
    else:
        state_recovery = np.nan

    return {
        "excess_persistence_no_vol": excess_nv,
        "state_recovery_ari": state_recovery,
        "sep_ratio": float(np.nanmax(mu[finite]) / np.nanmin(mu[finite])) if finite.sum() > 1 else np.nan,
        "monotone": bool(np.all(np.diff(mu[finite]) > 0)) if finite.sum() > 1 else False,
        "mean_self_transition": float(np.mean(np.diag(A))),
        "excess_persistence": excess,
        "n_states_used": int(len(set(states))),
        "state_entropy": float(-np.sum(stat * np.log(np.maximum(stat, 1e-12)))),
        "redundancy_acc": redundancy,
        "redundancy_base_rate": base_rate,
    }


def run_one(dgp_name: str, rep: int) -> List[dict]:
    cfg = load_config()
    cfg = replace(cfg, project=replace(cfg.project, seed=cfg.project.seed))
    rng = np.random.default_rng(1000 * rep + abs(hash(dgp_name)) % 1000)

    returns = DGPS[dgp_name](N_OBS, rng)
    true_state = _TRUE_STATE.get("last")
    raw = _frame_from_returns(returns, rng)
    data = build_feature_table(cfg, raw)
    features = [c for c in data.columns if not c.startswith("y_")]

    # align the latent path to the rows that survive feature warm-up
    if true_state is not None:
        pos = raw.index.get_indexer(data.index)
        true_state = np.asarray(true_state)[pos]

    diag = hmm_diagnostics(data, cfg, seed=cfg.project.seed, true_state=true_state)

    hmm_kwargs = dict(
        covariance_type=cfg.hmm.covariance_type, n_iter=cfg.hmm.n_iter,
        tol=cfg.hmm.tol, min_covar=cfg.hmm.min_covar,
    )
    cache = HMMCache(hmm_kwargs)

    rows: List[dict] = []
    for h in HORIZONS:
        target = f"y_rv_h{h}"
        ev = EvalSpec(cfg.evaluation.train_years, cfg.evaluation.test_years,
                      cfg.evaluation.step_years, embargo=h)
        try:
            gated = collect_oos_regime_nested(
                df=data, features=features, target=target, ev=ev,
                K_values=[int(k) for k in cfg.hmm.K_values], modes=["hard", "soft"],
                ablation=ABLATION_BY_NAME["normal"], cache=cache, seed=cfg.project.seed,
            )
            pooled = collect_oos_regime_nested(
                df=data, features=features, target=target, ev=ev,
                K_values=[int(k) for k in cfg.hmm.K_values], modes=["hard", "soft"],
                ablation=ABLATION_BY_NAME["single"], cache=cache, seed=cfg.project.seed,
            )
        except Exception as e:
            rows.append({"dgp": dgp_name, "rep": rep, "horizon": h,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        idx = gated.y_true.index.intersection(pooled.y_true.index)
        y = gated.y_true.loc[idx].to_numpy()
        r_gate = rmse(y, gated.y_pred.loc[idx].to_numpy())
        r_pool = rmse(y, pooled.y_pred.loc[idx].to_numpy())

        rows.append({
            "dgp": dgp_name,
            "has_true_regimes": DGP_HAS_REGIMES[dgp_name],
            "rep": rep,
            "horizon": h,
            "rmse_gated": r_gate,
            "rmse_pooled": r_pool,
            "improvement_pct": 100.0 * (r_pool - r_gate) / r_pool,
            "uncond_vol": float(data["ret_vol_20"].mean()),
            **diag,
        })

    return rows


def main(n_reps: int = 12, workers: int = 14) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(d, r) for d in DGPS for r in range(n_reps)]
    print(f"=== {len(jobs)} runs ({len(DGPS)} DGPs x {n_reps} reps) on {workers} workers ===",
          flush=True)

    all_rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, d, r): (d, r) for d, r in jobs}
        for fut in as_completed(futs):
            d, r = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[fail] {d} rep{r}: {type(e).__name__}: {e}", flush=True)
                continue
            all_rows.extend(rows)
            ok = [x for x in rows if "error" not in x]
            if ok:
                msg = "  ".join(f"h={x['horizon']}:{x['improvement_pct']:+.2f}%" for x in ok)
                print(f"[done] {d:<11} rep{r:<2} sep={ok[0]['sep_ratio']:.2f} "
                      f"persist={ok[0]['excess_persistence']:.3f}"
                      f"/{ok[0]['excess_persistence_no_vol']:+.3f} "
                      f"ari={ok[0]['state_recovery_ari']:.2f}  {msg}", flush=True)

    df = pd.DataFrame(all_rows)
    out = Path("artifacts") / "study" / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "simulation_results.csv", index=False)
    print(f"\nWrote {out / 'simulation_results.csv'}")
    return out


if __name__ == "__main__":
    main()
