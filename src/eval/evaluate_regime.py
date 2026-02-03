from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.eval.metrics import (
    mae,
    rmse,
    directional_accuracy,
    spearman_corr,
    top_decile_hit_rate,
)
from src.eval.subsets import high_vol_mask, top_quantile_mask, apply_mask
from src.eval.walk_forward import walk_forward_splits
from src.models.regime_conditioned import RegimeConditionedRidge
from src.regime.hmm import fit_hmm_and_infer_probs, hmm_interpretability


@dataclass(frozen=True)
class RegimeEvalConfig:
    K: int
    hmm_covariance_type: str
    hmm_n_iter: int
    hmm_tol: float
    hmm_min_covar: float
    seed: int


def evaluate_hmm_regime_ridge(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: RegimeConditionedRidge,
    train_years: int,
    test_years: int,
    step_years: int,
    regime_cfg: RegimeEvalConfig,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]], pd.DataFrame]:
    """
    walk-forward eval for regime-conditioned Ridge using HMM regime probs.

    for return targets: rmse, mae, directional_accuracy
    for volatility targets: rmse, mae, spearman, top_decile_hit_rate,
      plus stress subset metrics:
      - *_hv_now: top decile of ret_vol_20 within each test window (current stress)
      - *_hv_fut: top decile of y_true within each test window (future spike stress)
    """
    is_vol_target = ("_absret_" in target) or ("_sqret_" in target)

    if is_vol_target:
        metrics: Dict[str, List[float]] = {
            "rmse": [],
            "mae": [],
            "spearman": [],
            "top_decile_hit": [],
            "rmse_hv_now": [],
            "mae_hv_now": [],
            "rmse_hv_fut": [],
            "mae_hv_fut": [],
        }
    else:
        metrics = {
            "rmse": [],
            "mae": [],
            "directional_accuracy": [],
        }
    y_true_oos = []
    y_pred_oos = []
    last_hmm_res = None
    oos_rows: List[dict] = []

    for split in walk_forward_splits(df.index, train_years, test_years, step_years):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        hmm_res = fit_hmm_and_infer_probs(
            train_df=train,
            test_df=test,
            K=regime_cfg.K,
            covariance_type=regime_cfg.hmm_covariance_type,
            n_iter=regime_cfg.hmm_n_iter,
            tol=regime_cfg.hmm_tol,
            min_covar=regime_cfg.hmm_min_covar,
            seed=regime_cfg.seed,
        )

        last_hmm_res = hmm_res

        # Align probs to X/y lengths (HMM obs builder may drop NaNs)
        n_train = min(len(X_train), hmm_res.train_probs.shape[0])
        n_test = min(len(X_test), hmm_res.test_probs.shape[0])

        Xtr = X_train.iloc[-n_train:]
        ytr = y_train.iloc[-n_train:]
        Xte = X_test.iloc[-n_test:]
        yte = y_test.iloc[-n_test:]
        test_aligned = test.iloc[-n_test:]  # aligns with yte/y_pred for hv_now

        # collect strictly OOS regime probs aligned to Xte/yte
        probs_oos = hmm_res.test_probs[-n_test:]
        dates_oos = test_aligned.index
        for d, p in zip(dates_oos, probs_oos):
            row = {"date": d}
            for k in range(p.shape[0]):
                row[f"p_state_{k}"] = float(p[k])
            row["hard_state"] = int(np.argmax(p))
            row["max_prob"] = float(np.max(p))
            oos_rows.append(row)

        model.fit(Xtr, ytr, hmm_res.train_probs[-n_train:])
        y_pred = model.predict(Xte, hmm_res.test_probs[-n_test:])

        y_true_oos.append(pd.Series(yte, index=test_aligned.index))
        y_pred_oos.append(pd.Series(y_pred, index=test_aligned.index))

        # Full-sample metrics
        metrics["rmse"].append(rmse(yte, y_pred))
        metrics["mae"].append(mae(yte, y_pred))

        if is_vol_target:
            metrics["spearman"].append(spearman_corr(yte, y_pred))
            metrics["top_decile_hit"].append(top_decile_hit_rate(yte, y_pred))

            yt = np.asarray(yte, dtype=float)
            yp = np.asarray(y_pred, dtype=float)

            # hv_now: current stress defined by ret_vol_20 within THIS test window
            hv_now = high_vol_mask(test_aligned, vol_col="ret_vol_20", q=0.9)
            yt_now, yp_now = apply_mask(yt, yp, hv_now)
            if len(yt_now) > 0:
                metrics["rmse_hv_now"].append(rmse(yt_now, yp_now))
                metrics["mae_hv_now"].append(mae(yt_now, yp_now))

            # hv_fut: future stress defined by top decile of y_true within THIS test window
            hv_fut = top_quantile_mask(yt, q=0.9)
            yt_fut, yp_fut = apply_mask(yt, yp, hv_fut)
            if len(yt_fut) > 0:
                metrics["rmse_hv_fut"].append(rmse(yt_fut, yp_fut))
                metrics["mae_hv_fut"].append(mae(yt_fut, yp_fut))
        else:
            metrics["directional_accuracy"].append(directional_accuracy(yte, y_pred))
    
    hmm_info: Optional[Dict[str, np.ndarray]] = None
    if last_hmm_res is not None:
        hmm_info = hmm_interpretability(last_hmm_res.model, last_hmm_res.scaler)
    
    # build OOS regime probability df
    oos_df = pd.DataFrame(oos_rows)
    if len(oos_df) > 0:
        oos_df = oos_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        oos_df["date"] = pd.to_datetime(oos_df["date"])
        oos_df = oos_df.set_index("date")

    y_true_oos = pd.concat(y_true_oos).sort_index()
    y_pred_oos = pd.concat(y_pred_oos).sort_index()

    # safe aggregation
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(np.mean(v)) if len(v) else float("nan")
    return out, hmm_info, oos_df, y_true_oos, y_pred_oos
