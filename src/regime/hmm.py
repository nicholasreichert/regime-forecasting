from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class HMMResult:
    model: GaussianHMM
    scaler: StandardScaler
    train_probs: np.ndarray  # filtered, shape (n_train, K)
    test_probs: np.ndarray   # filtered, shape (n_test, K)


def _build_hmm_observations(df: pd.DataFrame) -> pd.DataFrame:
    # low dimensional emission observational vector
    obs = pd.DataFrame(index=df.index)
    obs["ret"] = df["ret_1d"].astype(float)
    obs["abs_ret"] = df["ret_1d"].abs().astype(float)
    if "ret_vol_20" in df.columns:
        obs["vol"] = df["ret_vol_20"].astype(float)
    return obs.dropna()


def _log_gaussian_diag_density(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    # compute log N(x | mean_k, diag(covar_k)) for all t, k
    T, D = X.shape
    K = means.shape[0]

    covars = np.asarray(covars)

    # If covars are full matrices, take the diagonal
    if covars.ndim == 3:
        # (K, D, D) -> (K, D)
        covars = np.diagonal(covars, axis1=1, axis2=2)

    if covars.shape != (K, D):
        raise ValueError(f"Expected covars shape {(K, D)} after diag, got {covars.shape}")

    covars = np.maximum(covars, 1e-8)

    log_det = np.sum(np.log(covars), axis=1)  # (K,)

    diff = X[:, None, :] - means[None, :, :]  # (T, K, D)
    quad = np.sum((diff * diff) / covars[None, :, :], axis=2)  # (T, K)

    log_norm = 0.5 * (D * np.log(2.0 * np.pi) + log_det)  # (K,)
    return -(log_norm[None, :] + 0.5 * quad)



def forward_filter_probs(
    X: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    log_emission: np.ndarray,
    init_alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict online filtering:
    - X: (T, D) (not used directly here; included for signature clarity)
    - startprob: (K,)
    - transmat: (K, K)
    - log_emission: (T, K) where log_emission[t,k] = log p(x_t | z_t=k)
    - init_alpha: optional initial belief over states (K,). If provided, overrides startprob.

    Returns:
      - probs: (T, K) filtered posteriors p(z_t | x_{1:t})
      - last_alpha: (K,) filtered posterior at final step
    """
    T, K = log_emission.shape
    logA = np.log(np.maximum(transmat, 1e-300))  # avoid log(0)

    if init_alpha is None:
        alpha0 = np.maximum(startprob, 1e-300)
        log_alpha = np.log(alpha0)
    else:
        init_alpha = np.maximum(init_alpha, 1e-300)
        log_alpha = np.log(init_alpha)

    probs = np.zeros((T, K), dtype=float)

    for t in range(T):
        # Predict step: logsumexp over previous states
        # log_pred[j] = log sum_i exp(log_alpha[i] + logA[i,j])
        log_pred = logsumexp(log_alpha[:, None] + logA, axis=0)  # (K,)

        # Update step with emission
        log_alpha = log_pred + log_emission[t]  # (K,)

        # Normalize to get probabilities
        log_alpha -= logsumexp(log_alpha)
        probs[t] = np.exp(log_alpha)

    return probs, probs[-1].copy()


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
    Fit HMM on train observations only.
    Then compute STRICT ONLINE filtered probabilities for:
      - train (p(z_t | x_{1:t}) on train)
      - test  (p(z_t | x_{1:t}) on test, warm-started from last train belief)
    """
    train_obs_df = _build_hmm_observations(train_df)
    test_obs_df = _build_hmm_observations(test_df)

    train_obs = train_obs_df.to_numpy()
    test_obs = test_obs_df.to_numpy()

    scaler = StandardScaler()
    train_obs_z = scaler.fit_transform(train_obs)
    test_obs_z = scaler.transform(test_obs)

    hmm = GaussianHMM(
        n_components=K,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        min_covar=float(min_covar),
        random_state=seed,
        init_params="stmc",
        params="stmc",
    )
    hmm.fit(train_obs_z)

    # Build log emission probabilities from fitted params
    if hmm.covariance_type != "diag":
        raise ValueError(
            f"Strict filtering implementation currently supports covariance_type='diag'. "
            f"Got {hmm.covariance_type}. Set config to diag."
        )

    logB_train = _log_gaussian_diag_density(train_obs_z, hmm.means_, hmm.covars_)
    logB_test = _log_gaussian_diag_density(test_obs_z, hmm.means_, hmm.covars_)

    # Strict online filtering
    train_probs, last_alpha = forward_filter_probs(
        X=train_obs_z,
        startprob=hmm.startprob_,
        transmat=hmm.transmat_,
        log_emission=logB_train,
        init_alpha=None,
    )

    # Warm-start test with last train belief (more realistic than resetting to startprob)
    test_probs, _ = forward_filter_probs(
        X=test_obs_z,
        startprob=hmm.startprob_,
        transmat=hmm.transmat_,
        log_emission=logB_test,
        init_alpha=last_alpha,
    )

    return HMMResult(model=hmm, scaler=scaler, train_probs=train_probs, test_probs=test_probs)

def hmm_interpretability(hmm, scaler=None):
    transmat = hmm.transmat_
    startprob = hmm.startprob_

    # stationary distribution
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    stat = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stat = stat[:, 0]
    stat = stat / stat.sum()

    means = hmm.means_
    covars = hmm.covars_

    if scaler is not None:
        means_raw = scaler.inverse_transform(means)
    else:
        means_raw = means

    return {
        "transmat": transmat,
        "startprob": startprob,
        "stationary": stat,
        "means_z": means,
        "means_raw": means_raw,
        "covars": covars,
    }
