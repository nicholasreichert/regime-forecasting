from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Default Wilder/RiskMetrics decay for daily data.
RISKMETRICS_LAMBDA = 0.94


class BaselineModel:
    """Common interface: ``fit(X, y)`` then ``predict(X)``.

    ``X`` is the full feature frame, so models are free to pick out the columns
    they need (``ret_1d`` for the conditional-variance models, ``rv_bwd_*`` for
    HAR). ``params`` is recorded in the results log.
    """

    name: str = "baseline"

    @property
    def params(self) -> dict:
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# trivial references
# --------------------------------------------------------------------------


class ZeroReturnBaseline(BaselineModel):
    name = "zero_return"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RollingMeanBaseline(BaselineModel):
    """Unconditional mean of the training target (a constant forecast)."""

    name = "train_mean"

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class RandomWalkVolBaseline(BaselineModel):
    """Random walk in realized volatility: forecast RV_{t+1:t+h} with RV_{t-w+1:t}.

    The hardest baseline to beat at short horizons in most RV studies.
    """

    def __init__(self, rv_col: str = "rv_bwd_22"):
        self.rv_col = rv_col
        self.name = f"rw_vol[{rv_col}]"

    @property
    def params(self) -> dict:
        return {"rv_col": self.rv_col}

    def fit(self, X, y):
        if self.rv_col not in X.columns:
            raise KeyError(f"RandomWalkVolBaseline needs column '{self.rv_col}'")
        return self

    def predict(self, X):
        return X[self.rv_col].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# linear models
# --------------------------------------------------------------------------


class RidgeBaseline(BaselineModel):
    """Ridge on the full feature set, standardized, with alpha tuned in-sample.

    Standardization matters here: the raw features span roughly three orders of
    magnitude (log returns ~1e-3, RSI ~1e2), so a fixed penalty on unscaled
    inputs shrinks them wildly unevenly. Alpha is chosen by leave-one-out
    generalized cross-validation on the *training* window only.
    """

    def __init__(self, alpha: Optional[float] = None, alphas: Sequence[float] | None = None):
        self.alpha = alpha
        self.alphas = tuple(alphas) if alphas is not None else tuple(np.logspace(-3, 4, 24))
        self.name = "ridge" if alpha is None else f"ridge[a={alpha:g}]"

    @property
    def params(self) -> dict:
        return {"alpha": self.alpha, "tuned": self.alpha is None, "chosen_alpha": getattr(self, "chosen_alpha_", None)}

    def fit(self, X, y):
        if self.alpha is None:
            reg = RidgeCV(alphas=self.alphas)
        else:
            reg = Ridge(alpha=self.alpha)
        self.model_ = Pipeline([("scale", StandardScaler()), ("reg", reg)])
        self.model_.fit(X, y)
        self.chosen_alpha_ = float(getattr(self.model_.named_steps["reg"], "alpha_", self.alpha or np.nan))
        return self

    def predict(self, X):
        return self.model_.predict(X)


class HARBaseline(BaselineModel):
    """Heterogeneous Autoregressive model of realized volatility (Corsi, 2009).

    RV_{t+1:t+h} = b0 + b_d RV_t^{(1)} + b_w RV_t^{(5)} + b_m RV_t^{(22)} + e

    The reference benchmark in the realized-volatility literature. Estimated by
    OLS (implemented as a ridge with a negligible penalty for numerical
    stability on collinear RV components).
    """

    def __init__(self, rv_cols: Sequence[str] = ("rv_bwd_1", "rv_bwd_5", "rv_bwd_22")):
        self.rv_cols = list(rv_cols)
        self.name = "har"

    @property
    def params(self) -> dict:
        return {"rv_cols": self.rv_cols, "coef": getattr(self, "coef_", None)}

    def fit(self, X, y):
        missing = [c for c in self.rv_cols if c not in X.columns]
        if missing:
            raise KeyError(f"HARBaseline missing columns: {missing}")
        self.model_ = Ridge(alpha=1e-8, fit_intercept=True)
        self.model_.fit(X[self.rv_cols], y)
        self.coef_ = [float(c) for c in np.atleast_1d(self.model_.coef_)]
        return self

    def predict(self, X):
        return self.model_.predict(X[self.rv_cols])


# --------------------------------------------------------------------------
# conditional-variance models
# --------------------------------------------------------------------------


def _fit_level_map(sigma_pred_train: np.ndarray, y_train: np.ndarray) -> tuple[float, float]:
    """OLS map from a volatility forecast onto the target's own scale.

    The conditional-variance models produce sqrt(E[sigma^2]) whereas the target
    is E[RV]; by Jensen these differ, and the gap is horizon-dependent. Fitting
    ``y ~ a + b * sigma_hat`` on the training window removes that systematic
    bias so the baseline competes on information content rather than on
    calibration. Estimated on training data only.
    """
    s = np.asarray(sigma_pred_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 10 or np.std(s[ok]) < 1e-12:
        return float(np.mean(y[ok])) if ok.any() else 0.0, 0.0
    A = np.column_stack([np.ones(ok.sum()), s[ok]])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return float(coef[0]), float(coef[1])


class EWMAVolBaseline(BaselineModel):
    """RiskMetrics EWMA variance forecast.

    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}

    EWMA is IGARCH(1,1) without drift, so the multi-step variance forecast is
    flat: E[sigma^2_{t+k} | F_t] = sigma^2_{t+1} for all k. The recursion is run
    causally through the test window using only realized returns up to t.
    """

    def __init__(self, horizon: int, lam: float = RISKMETRICS_LAMBDA, ret_col: str = "ret_1d"):
        self.horizon = int(horizon)
        self.lam = float(lam)
        self.ret_col = ret_col
        self.name = "ewma"

    @property
    def params(self) -> dict:
        return {"lambda": self.lam, "horizon": self.horizon, "a": getattr(self, "a_", None), "b": getattr(self, "b_", None)}

    def _sigma_path(self, r: np.ndarray, sigma2_0: float) -> np.ndarray:
        """One-step-ahead sigma forecasts: element t is E[sigma_{t+1} | r_{1..t}]."""
        n = len(r)
        out = np.empty(n, dtype=float)
        s2 = float(sigma2_0)
        for t in range(n):
            # variance forecast for t+1 given returns through t
            s2 = self.lam * s2 + (1.0 - self.lam) * float(r[t]) ** 2
            out[t] = s2
        return np.sqrt(out)

    def fit(self, X, y):
        r = X[self.ret_col].to_numpy(dtype=float)
        self.sigma2_init_ = float(np.mean(r**2))
        sigma_train = self._sigma_path(r, self.sigma2_init_)
        # carry the filter state forward into the test window
        self.sigma2_last_ = float(sigma_train[-1] ** 2)
        self.a_, self.b_ = _fit_level_map(sigma_train, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        r = X[self.ret_col].to_numpy(dtype=float)
        sigma = self._sigma_path(r, self.sigma2_last_)
        return self.a_ + self.b_ * sigma


class GARCHBaseline(BaselineModel):
    """GARCH(1,1) with Gaussian errors, fitted on the training window.

    Parameters (omega, alpha, beta) are estimated once per training window; the
    conditional-variance recursion is then filtered forward through the test
    window using only realized returns up to t, so no test information enters
    the parameters or the state. The h-step aggregate forecast uses the standard
    mean-reversion formula

        E[sigma^2_{t+k}] = s2_inf + (alpha + beta)^{k-1} (sigma^2_{t+1} - s2_inf)

    aggregated over k = 1..h.
    """

    def __init__(self, horizon: int, ret_col: str = "ret_1d"):
        self.horizon = int(horizon)
        self.ret_col = ret_col
        self.name = "garch11"

    @property
    def params(self) -> dict:
        return {
            "horizon": self.horizon,
            "omega": getattr(self, "omega_", None),
            "alpha": getattr(self, "alpha_", None),
            "beta": getattr(self, "beta_", None),
            "converged": getattr(self, "converged_", None),
        }

    def _aggregate_sigma(self, r: np.ndarray, sigma2_0: float) -> np.ndarray:
        """Return sqrt( (1/h) * sum_{k=1..h} E[sigma^2_{t+k} | F_t] ) for each t."""
        h = self.horizon
        persist = self.alpha_ + self.beta_
        s2_inf = self.omega_ / max(1.0 - persist, 1e-8)

        # geometric aggregation weight: (1/h) * sum_{k=1..h} persist^{k-1}
        if abs(persist - 1.0) < 1e-10:
            agg_w = 1.0
        else:
            agg_w = (1.0 - persist**h) / (h * (1.0 - persist))

        n = len(r)
        out = np.empty(n, dtype=float)
        s2 = float(sigma2_0)
        for t in range(n):
            # one-step-ahead variance for t+1 given returns through t
            s2 = self.omega_ + self.alpha_ * float(r[t]) ** 2 + self.beta_ * s2
            out[t] = s2_inf + agg_w * (s2 - s2_inf)
        return np.sqrt(np.maximum(out, 1e-16))

    def fit(self, X, y):
        from arch import arch_model

        r = X[self.ret_col].to_numpy(dtype=float)
        # arch works better on percentage returns; rescale and undo afterwards.
        scale = 100.0
        am = arch_model(r * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)

        p = res.params
        self.omega_ = float(p["omega"]) / scale**2
        self.alpha_ = float(p["alpha[1]"])
        self.beta_ = float(p["beta[1]"])
        self.converged_ = bool(getattr(res, "convergence_flag", 0) == 0)

        self.sigma2_init_ = float(np.mean(r**2))
        sigma_train = self._aggregate_sigma(r, self.sigma2_init_)
        self.sigma2_last_ = float(res.conditional_volatility[-1] / scale) ** 2
        self.a_, self.b_ = _fit_level_map(sigma_train, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        r = X[self.ret_col].to_numpy(dtype=float)
        sigma = self._aggregate_sigma(r, self.sigma2_last_)
        return self.a_ + self.b_ * sigma
