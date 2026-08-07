from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.metrics import mae, qlike, rmse, spearman_corr, top_decile_hit_rate
from src.eval.walk_forward import Split, walk_forward_splits
from src.models.regime_conditioned import RegimeConditionedRidge
from src.regime.hmm import fit_hmm_and_infer_probs, hmm_interpretability

# --------------------------------------------------------------------------
# ablation design
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ablation:
    """How the regime probabilities fed to the regressor are perturbed.

    The expert *partition* (which rows train which expert) and the
    prediction-time *gate* are perturbed independently. This separates two
    distinct claims that a single "shuffle" ablation conflates: that
    regime-specialised experts are worth fitting, and that knowing the current
    regime at prediction time is worth anything.
    """

    name: str
    train: str  # "true" | "shuffle" | "single"
    test: str  # "true" | "shuffle" | "uniform" | "single"


ABLATIONS: tuple[Ablation, ...] = (
    Ablation("normal", "true", "true"),
    # experts trained on real regimes, but the gate is given the right marginal
    # regime mix in the wrong temporal order => isolates the value of timing.
    Ablation("gate_shuffle", "true", "shuffle"),
    # gate is real, but experts were fitted on an arbitrary partition of the
    # training window => isolates the value of regime-specific experts.
    Ablation("expert_shuffle", "shuffle", "true"),
    # both perturbed: the original single-shuffle control.
    Ablation("both_shuffle", "shuffle", "shuffle"),
    # collapses to a single pooled ridge; the control arm.
    Ablation("single", "single", "single"),
)

ABLATION_BY_NAME = {a.name: a for a in ABLATIONS}


def _apply_prob_transform(probs: np.ndarray, how: str, rng: np.random.Generator) -> np.ndarray:
    T, K = probs.shape
    if how == "true":
        return probs
    if how == "shuffle":
        return probs[rng.permutation(T)]
    if how == "uniform":
        return np.full((T, K), 1.0 / K, dtype=float)
    if how == "single":
        out = np.zeros((T, K), dtype=float)
        out[:, 0] = 1.0
        return out
    raise ValueError(f"Unknown probability transform: {how}")


# --------------------------------------------------------------------------
# out-of-sample prediction collection
# --------------------------------------------------------------------------


@dataclass
class OOSResult:
    y_true: pd.Series
    y_pred: pd.Series
    regime_probs: Optional[pd.DataFrame] = None
    hmm_info: Optional[Dict[str, np.ndarray]] = None
    fold_boundaries: Optional[List[pd.Timestamp]] = None
    diagnostics: Optional[Dict[str, float]] = None


def _splits(df: pd.DataFrame, ev: "EvalSpec") -> List[Split]:
    return list(
        walk_forward_splits(
            df.index,
            train_years=ev.train_years,
            test_years=ev.test_years,
            step_years=ev.step_years,
            embargo=ev.embargo,
        )
    )


@dataclass(frozen=True)
class EvalSpec:
    train_years: int
    test_years: int
    step_years: int
    embargo: int  # set to the target horizon h


def collect_oos_baseline(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    model_factory: Callable[[], object],
    ev: EvalSpec,
) -> OOSResult:
    """Walk-forward OOS predictions for a plain ``fit(X, y)`` / ``predict(X)`` model."""
    y_true_parts: List[pd.Series] = []
    y_pred_parts: List[pd.Series] = []
    boundaries: List[pd.Timestamp] = []

    for split in _splits(df, ev):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        model = model_factory()
        model.fit(train[list(features)], train[target])
        pred = np.asarray(model.predict(test[list(features)]), dtype=float)

        y_true_parts.append(test[target].astype(float))
        y_pred_parts.append(pd.Series(pred, index=test.index))
        boundaries.append(split.test_start)

    if not y_true_parts:
        raise RuntimeError("No walk-forward splits were produced; check the date range.")

    return OOSResult(
        y_true=pd.concat(y_true_parts).sort_index(),
        y_pred=pd.concat(y_pred_parts).sort_index(),
        fold_boundaries=boundaries,
    )


def collect_oos_regime(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    ev: EvalSpec,
    K: int,
    mode: str,
    ablation: Ablation,
    hmm_kwargs: dict,
    ridge_alpha: Optional[float] = None,
    min_points_per_regime: int = 200,
    seed: int = 0,
) -> OOSResult:
    """Walk-forward OOS predictions for the HMM-gated regime-conditioned ridge.

    The HMM is refitted on each training window and regime probabilities on the
    test window are produced by strict online filtering (warm-started from the
    final training belief), so nothing about the test period enters either the
    HMM parameters or the filtered state.
    """
    y_true_parts: List[pd.Series] = []
    y_pred_parts: List[pd.Series] = []
    prob_parts: List[pd.DataFrame] = []
    boundaries: List[pd.Timestamp] = []
    last_hmm = None
    n_experts: List[int] = []

    for split_idx, split in enumerate(_splits(df, ev)):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        hmm_res = fit_hmm_and_infer_probs(
            train_df=train, test_df=test, K=K, seed=seed, **hmm_kwargs
        )
        last_hmm = hmm_res

        n_train = min(len(train), hmm_res.train_probs.shape[0])
        n_test = min(len(test), hmm_res.test_probs.shape[0])

        Xtr = train[list(features)].iloc[-n_train:]
        ytr = train[target].iloc[-n_train:]
        te = test.iloc[-n_test:]
        Xte = te[list(features)]
        yte = te[target]

        probs_tr_true = hmm_res.train_probs[-n_train:]
        probs_te_true = hmm_res.test_probs[-n_test:]

        # record the *unperturbed* filtered probabilities for diagnostics
        pcols = {f"p_state_{k}": probs_te_true[:, k] for k in range(probs_te_true.shape[1])}
        pdf = pd.DataFrame(pcols, index=te.index)
        pdf["hard_state"] = probs_te_true.argmax(axis=1)
        pdf["max_prob"] = probs_te_true.max(axis=1)
        prob_parts.append(pdf)

        rng_tr = np.random.default_rng(seed + 10_000 + split_idx)
        rng_te = np.random.default_rng(seed + 20_000 + split_idx)
        probs_tr = _apply_prob_transform(probs_tr_true, ablation.train, rng_tr)
        probs_te = _apply_prob_transform(probs_te_true, ablation.test, rng_te)

        model = RegimeConditionedRidge(
            alpha=ridge_alpha, mode=mode, min_points_per_regime=min_points_per_regime
        )
        model.fit(Xtr, ytr, probs_tr)
        pred = model.predict(Xte, probs_te)
        n_experts.append(int(getattr(model, "n_experts_", 0)))

        y_true_parts.append(yte.astype(float))
        y_pred_parts.append(pd.Series(np.asarray(pred, dtype=float), index=te.index))
        boundaries.append(split.test_start)

    if not y_true_parts:
        raise RuntimeError("No walk-forward splits were produced; check the date range.")

    hmm_info = hmm_interpretability(last_hmm.model, last_hmm.scaler) if last_hmm else None
    probs_df = pd.concat(prob_parts).sort_index()
    probs_df = probs_df[~probs_df.index.duplicated(keep="last")]

    return OOSResult(
        y_true=pd.concat(y_true_parts).sort_index(),
        y_pred=pd.concat(y_pred_parts).sort_index(),
        regime_probs=probs_df,
        hmm_info=hmm_info,
        fold_boundaries=boundaries,
        diagnostics={"mean_experts_fitted": float(np.mean(n_experts)) if n_experts else np.nan},
    )


# --------------------------------------------------------------------------
# nested (leak-free) selection of K and gating mode
# --------------------------------------------------------------------------


class HMMCache:
    """Memoises HMM fits, which dominate the cost of the study.

    A fit depends only on (training rows, test rows, K, seed) - not on the
    target, the ablation, or the gating mode - so the same fit is reused across
    every configuration that shares those.
    """

    def __init__(self, hmm_kwargs: dict):
        self.hmm_kwargs = hmm_kwargs
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, train_df: pd.DataFrame, test_df: pd.DataFrame, K: int, seed: int):
        key = (train_df.index[0], train_df.index[-1], len(train_df),
               test_df.index[0], test_df.index[-1], len(test_df), int(K), int(seed))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        res = fit_hmm_and_infer_probs(
            train_df=train_df, test_df=test_df, K=int(K), seed=int(seed), **self.hmm_kwargs
        )
        self._cache[key] = res
        return res


def collect_oos_regime_nested(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    ev: EvalSpec,
    K_values: Sequence[int],
    modes: Sequence[str],
    ablation: Ablation,
    cache: HMMCache,
    ridge_alpha: Optional[float] = None,
    min_points_per_regime: int = 200,
    seed: int = 0,
    val_fraction: float = 0.25,
    selection_metric: str = "rmse",
) -> OOSResult:
    """As :func:`collect_oos_regime`, but K and the gating mode are chosen per
    fold on an inner validation block carved from the *training* window.

    This is the difference between reporting the best out-of-sample number over
    a grid (which is a biased estimate of what the method would have achieved in
    real time) and reporting what a forecaster restricted to past information
    would actually have obtained.
    """
    from src.eval.walk_forward import inner_validation_split

    y_true_parts: List[pd.Series] = []
    y_pred_parts: List[pd.Series] = []
    prob_parts: List[pd.DataFrame] = []
    boundaries: List[pd.Timestamp] = []
    chosen: List[tuple[int, str]] = []
    last_hmm = None

    for split_idx, split in enumerate(_splits(df, ev)):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        # ---- inner selection -------------------------------------------
        inner_tr_idx, inner_val_idx = inner_validation_split(
            split.train_idx, val_fraction=val_fraction, embargo=ev.embargo
        )
        inner_tr = df.loc[inner_tr_idx]
        inner_val = df.loc[inner_val_idx]

        best_key: tuple[int, str] | None = None
        best_score = np.inf
        for K in K_values:
            hres = cache.get(inner_tr, inner_val, K, seed)
            n_tr = min(len(inner_tr), hres.train_probs.shape[0])
            n_va = min(len(inner_val), hres.test_probs.shape[0])
            Xtr_i = inner_tr[list(features)].iloc[-n_tr:]
            ytr_i = inner_tr[target].iloc[-n_tr:]
            Xva_i = inner_val[list(features)].iloc[-n_va:]
            yva_i = inner_val[target].iloc[-n_va:].to_numpy(dtype=float)

            rng_tr = np.random.default_rng(seed + 30_000 + split_idx)
            rng_va = np.random.default_rng(seed + 40_000 + split_idx)
            ptr = _apply_prob_transform(hres.train_probs[-n_tr:], ablation.train, rng_tr)
            pva = _apply_prob_transform(hres.test_probs[-n_va:], ablation.test, rng_va)

            for mode in modes:
                m = RegimeConditionedRidge(
                    alpha=ridge_alpha, mode=mode, min_points_per_regime=min_points_per_regime
                )
                m.fit(Xtr_i, ytr_i, ptr)
                pv = m.predict(Xva_i, pva)
                score = rmse(yva_i, pv) if selection_metric == "rmse" else mae(yva_i, pv)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_key = (int(K), mode)

        if best_key is None:  # degenerate fold; fall back to the smallest model
            best_key = (int(min(K_values)), modes[0])
        chosen.append(best_key)
        K_star, mode_star = best_key

        # ---- refit on the full training window and predict the test fold --
        hres = cache.get(train, test, K_star, seed)
        last_hmm = hres

        n_train = min(len(train), hres.train_probs.shape[0])
        n_test = min(len(test), hres.test_probs.shape[0])

        Xtr = train[list(features)].iloc[-n_train:]
        ytr = train[target].iloc[-n_train:]
        te = test.iloc[-n_test:]

        probs_tr_true = hres.train_probs[-n_train:]
        probs_te_true = hres.test_probs[-n_test:]

        pcols = {f"p_state_{k}": probs_te_true[:, k] for k in range(probs_te_true.shape[1])}
        pdf = pd.DataFrame(pcols, index=te.index)
        pdf["hard_state"] = probs_te_true.argmax(axis=1)
        pdf["max_prob"] = probs_te_true.max(axis=1)
        pdf["K"] = K_star
        prob_parts.append(pdf)

        rng_tr = np.random.default_rng(seed + 10_000 + split_idx)
        rng_te = np.random.default_rng(seed + 20_000 + split_idx)
        probs_tr = _apply_prob_transform(probs_tr_true, ablation.train, rng_tr)
        probs_te = _apply_prob_transform(probs_te_true, ablation.test, rng_te)

        model = RegimeConditionedRidge(
            alpha=ridge_alpha, mode=mode_star, min_points_per_regime=min_points_per_regime
        )
        model.fit(Xtr, ytr, probs_tr)
        pred = model.predict(te[list(features)], probs_te)

        y_true_parts.append(te[target].astype(float))
        y_pred_parts.append(pd.Series(np.asarray(pred, dtype=float), index=te.index))
        boundaries.append(split.test_start)

    hmm_info = hmm_interpretability(last_hmm.model, last_hmm.scaler) if last_hmm else None
    probs_df = pd.concat(prob_parts).sort_index()
    probs_df = probs_df[~probs_df.index.duplicated(keep="last")]

    ks = [c[0] for c in chosen]
    soft_frac = float(np.mean([c[1] == "soft" for c in chosen])) if chosen else np.nan

    return OOSResult(
        y_true=pd.concat(y_true_parts).sort_index(),
        y_pred=pd.concat(y_pred_parts).sort_index(),
        regime_probs=probs_df,
        hmm_info=hmm_info,
        fold_boundaries=boundaries,
        diagnostics={
            "sel_K_mean": float(np.mean(ks)) if ks else np.nan,
            "sel_K_mode": float(max(set(ks), key=ks.count)) if ks else np.nan,
            "sel_soft_frac": soft_frac,
        },
    )


# --------------------------------------------------------------------------
# metrics over pooled OOS predictions
# --------------------------------------------------------------------------


def compute_metrics(
    res: OOSResult,
    df: pd.DataFrame,
    vol_col: str = "ret_vol_20",
    stress_q: float = 0.9,
) -> Dict[str, float]:
    """Metrics over the *pooled* out-of-sample period.

    Pooling (rather than averaging per-fold RMSEs) keeps every observation
    equally weighted and gives a single loss series per model, which is what the
    Diebold-Mariano tests operate on.

    Stress subsets:
      ``hv_now`` - top decile of trailing realized volatility, i.e. stress that
      is already visible at prediction time.
      ``hv_fut`` - top decile of the realized target, i.e. stress that has yet to
      materialise. This subset is defined by the outcome and so is only ever
      used for reporting, never for fitting or selection.
    """
    yt = res.y_true.to_numpy(dtype=float)
    yp = res.y_pred.to_numpy(dtype=float)

    out: Dict[str, float] = {
        "n_oos": int(len(yt)),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "qlike": qlike(yt, yp),
        "spearman": spearman_corr(yt, yp),
        "top_decile_hit": top_decile_hit_rate(yt, yp),
    }

    vol = df.loc[res.y_true.index, vol_col].to_numpy(dtype=float)
    hv_now = vol >= np.nanquantile(vol, stress_q)
    if hv_now.sum() > 0:
        out["rmse_hv_now"] = rmse(yt[hv_now], yp[hv_now])
        out["mae_hv_now"] = mae(yt[hv_now], yp[hv_now])

    hv_fut = yt >= np.nanquantile(yt, stress_q)
    if hv_fut.sum() > 0:
        out["rmse_hv_fut"] = rmse(yt[hv_fut], yp[hv_fut])
        out["mae_hv_fut"] = mae(yt[hv_fut], yp[hv_fut])

    if res.diagnostics:
        out.update(res.diagnostics)

    return out
