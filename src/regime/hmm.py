from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class HMMResult:
    model: GaussianHMM
    scaler: StandardScaler
    train_probs: np.ndarray     # shape: (n_train, K)
    test_probs: np.ndarray      # shape: (n_test, K)

def _build_hmm_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    creates a low-dim observation vector for HMM emissions
    
    requires ret_1d & ret_vol_* (for whatever computed)
    """
    required = ["ret_1d"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for HMM observations: {c}")
        
    # use ret_1d & abs(ret_1d), adding volatility feature
    obs = pd.DataFrame(index=df.index)
    obs["ret"] = df["ret_1d"].astype(float)
    obs["abs_ret"] = df["ret_1d"].abs().astype(float)

    vol_candidates = [c for c in df.columns if c.startswith("ret_vol_")]
    if "ret_vol_20" in df.columns:
        obs["vol"] = df["ret_vol_20"].astype(float)
    elif len(vol_candidates) > 0:
        obs["vol"] = df[sorted(vol_candidates)[-1]].astype(float)   # highest window
    else:
        pass

    obs = obs.dropna()
    return obs

def fit_hmm_and_infer_probs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    K: int,
    covariance_type: str = "diag",
    n_iter: int = 400,
    tol: float = 1e-4,
    min_covar: float = 1e-3,
    seed: int = 42,
) -> HMMResult:
    """
    fit hmm on train observations only, then compute filtered state probs
    (posterior) for both train and test

    standardize base don train only to avoid leakage*
    """
    train_obs_df = _build_hmm_observations(train_df)
    test_obs_df = _build_hmm_observations(test_df)

    train_obs = train_obs_df.to_numpy()
    test_obs = test_obs_df.to_numpy()

    scaler = StandardScaler()
    train_obs_z = scaler.fit_transform(train_obs)
    test_obs_z = scaler.transform(test_obs)

    def _fit(cov_type: str) -> GaussianHMM:
        hmm = GaussianHMM(
            n_components=K,
            covariance_type=cov_type,
            n_iter=n_iter,
            tol=tol,
            min_covar=min_covar,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )
        hmm.fit(train_obs_z)
        return hmm
    
    try:
        hmm = _fit(covariance_type)
    except ValueError as e:
        if covariance_type != "diag":
            hmm = _fit("diag")
        else:
            raise e
        
    train_probs = hmm.predict_proba(train_obs_z)
    test_probs = hmm.predict_proba(test_obs_z)

    return HMMResult(model=hmm, scaler=scaler, train_probs=train_probs, test_probs=test_probs)

