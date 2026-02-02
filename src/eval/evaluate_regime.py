from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval.metrics import mae, rmse, directional_accuracy
from src.eval.walk_forward import walk_forward_splits
from src.regime.hmm import fit_hmm_and_infer_probs
from src.models.regime_conditioned import RegimeConditionedRidge


@dataclass(frozen=True)
class RegimeEvalConfig:
    K: int
    hmm_covariance_type: str
    hmm_n_iter: int
    hmm_tol: float
    hmm_min_covar = 1e-3
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
) -> Dict[str, float]:
    metrics = {"rmse": [], "mae": [], "directional_accuracy": []}

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

        # align probs to X/y lengths (since HMM observation builder drops NaNs)
        n_train = min(len(X_train), hmm_res.train_probs.shape[0])
        n_test = min(len(X_test), hmm_res.test_probs.shape[0])

        model.fit(X_train.iloc[-n_train:], y_train.iloc[-n_train:], hmm_res.train_probs[-n_train:])
        y_pred = model.predict(X_test.iloc[-n_test:], hmm_res.test_probs[-n_test:])

        y_true = y_test.iloc[-n_test:]

        metrics["rmse"].append(rmse(y_true, y_pred))
        metrics["mae"].append(mae(y_true, y_pred))
        metrics["directional_accuracy"].append(directional_accuracy(y_true, y_pred))

    return {k: float(np.mean(v)) for k, v in metrics.items()}