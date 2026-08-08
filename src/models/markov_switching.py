"""Markov-switching regression: states estimated *jointly* with the forecast.

The two-stage pipeline this paper evaluates fits an HMM to returns and then
conditions a separate regressor on the inferred state. Section
``sec:simulation`` shows why that is handicapped: Baum-Welch maximises the
likelihood of the *emissions*, so it identifies states that differ in their
marginal distribution, and is close to blind to states that differ in how the
predictors map to the target. Those are exactly the states worth conditioning on.

This module implements the alternative the diagnosis points at. The emission
*is* the predictive density,

    p(y_t | x_t, z_t = k) = N(beta_k' x_t, sigma_k^2),

with Markov transitions on z. Estimation is by EM on that joint likelihood, so
the states are defined by the regression relationship rather than by the shape
of the feature distribution.

This is Hamilton's (1989) Markov-switching regression; nothing here is a new
estimator. What is new in this paper is the comparison: the same data, the same
protocol, and a controlled simulation in which the two-stage pipeline provably
fails and joint estimation is expected not to.

Out-of-sample prediction is strictly causal and accounts for the forecast
horizon. The target at time t is not observed until t+h, so the filter is
updated only with targets that have actually been revealed and the state
distribution is then propagated forward the remaining steps via A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_LOG_2PI = float(np.log(2.0 * np.pi))


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float):
    """Weighted ridge with an unpenalised intercept, via the normal equations.

    Equivalent to ``Ridge(alpha).fit(X, y, sample_weight=w)``: centre by the
    weighted means, solve the penalised system for the slopes, then recover the
    intercept. Matches sklearn to numerical precision (asserted in the tests)
    and avoids its per-call overhead in the EM inner loop.
    """
    sw = w.sum()
    xbar = (w @ X) / sw
    ybar = float((w @ y) / sw)
    Xc = X - xbar
    yc = y - ybar

    Xw = Xc * w[:, None]
    A = Xc.T @ Xw
    A.flat[:: A.shape[0] + 1] += alpha  # add alpha to the diagonal
    b = Xw.T @ yc
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta, float(ybar - xbar @ beta)


@dataclass
class MarkovSwitchingRegression:
    """EM-estimated mixture of linear experts with a Markov gate.

    ``alpha`` is a ridge penalty applied to each state's weighted least squares
    update. Some regularisation is essential: each state is fitted on an
    effective sample of ``sum_t gamma_t(k)`` rows, which for a rarely-visited
    state can be far smaller than the feature count.
    """

    K: int = 2
    alpha: float = 1.0
    n_iter: int = 100
    tol: float = 1e-5
    min_var: float = 1e-12
    seed: int = 0
    n_init: int = 3  # EM is non-convex; keep the best of several starts

    converged_: bool = field(default=False, init=False)
    loglik_: float = field(default=-np.inf, init=False)

    # ---------------- internals ----------------

    def _log_emission(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """log p(y_t | x_t, z_t=k) for every t, k."""
        resid = y[:, None] - X @ self.beta_.T - self.intercept_[None, :]
        return -0.5 * (_LOG_2PI + np.log(self.var_)[None, :] + resid**2 / self.var_[None, :])

    @staticmethod
    def _forward_backward(log_b: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
        """Scaled (Rabiner) forward-backward.

        The log-space version calls ``logsumexp`` twice per time step, which for
        T of a few thousand and up to ``n_init * n_iter`` EM passes dominates the
        entire fit. The scaled recursions normalise at each step instead, so the
        quantities stay O(1) without underflow and each step is a couple of small
        matrix products. Numerically equivalent; asserted against the log-space
        implementation in the tests.
        """
        T, K = log_b.shape
        # Work with emissions normalised per time step; the per-step maximum is
        # folded back into the log-likelihood so nothing is lost.
        m = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - m)
        A = np.exp(log_A)
        pi = np.exp(log_pi)

        alpha = np.empty((T, K))
        scale = np.empty(T)

        a = pi * b[0]
        s = a.sum()
        if s <= 0 or not np.isfinite(s):
            a, s = np.full(K, 1.0 / K), 1.0
        alpha[0], scale[0] = a / s, s
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * b[t]
            s = a.sum()
            if s <= 0 or not np.isfinite(s):
                a, s = np.full(K, 1.0 / K), 1.0
            alpha[t], scale[t] = a / s, s

        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (b[t + 1] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        # xi_t(i,j) ∝ alpha_t(i) A_ij b_{t+1}(j) beta_{t+1}(j) / scale_{t+1}
        bb = (b[1:] * beta[1:]) / scale[1:, None]           # (T-1, K)
        xi = np.einsum("ti,ij,tj->ij", alpha[:-1], A, bb)

        ll = float(np.sum(np.log(scale)) + m.sum())
        return gamma, xi, ll

    def _fit_once(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        T, D = X.shape
        K = self.K

        # initialise from a global fit perturbed per state, plus a sticky chain
        base = Ridge(alpha=self.alpha).fit(X, y)
        scale = np.std(y - base.predict(X)) + 1e-12
        self.beta_ = np.repeat(base.coef_[None, :], K, axis=0) * (
            1.0 + 0.5 * rng.normal(size=(K, D))
        )
        self.intercept_ = np.full(K, float(base.intercept_)) + 0.5 * scale * rng.normal(size=K)
        self.var_ = np.full(K, scale**2) * np.exp(rng.normal(scale=0.3, size=K))
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9)
        self.transmat_ = A
        self.startprob_ = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            log_A = np.log(np.maximum(self.transmat_, 1e-300))
            log_pi = np.log(np.maximum(self.startprob_, 1e-300))
            gamma, xi, ll = self._forward_backward(self._log_emission(X, y), log_pi, log_A)

            # M-step: weighted ridge per state, solved directly.
            # sklearn's Ridge is fine but is called K times per EM iteration for
            # up to n_init * n_iter iterations, where its per-call overhead
            # dominates the actual solve on a matrix this size.
            for k in range(K):
                w = gamma[:, k]
                sw = w.sum()
                if sw < 2 * self.K:  # state has collapsed; leave it alone
                    continue
                b, c = _weighted_ridge(X, y, w, self.alpha)
                self.beta_[k] = b
                self.intercept_[k] = c
                r = y - (X @ b + c)
                self.var_[k] = max(float(np.dot(w, r**2) / sw), self.min_var)

            denom = xi.sum(axis=1, keepdims=True)
            self.transmat_ = np.where(denom > 0, xi / np.maximum(denom, 1e-300),
                                      1.0 / K)
            self.startprob_ = np.maximum(gamma[0], 1e-12)
            self.startprob_ /= self.startprob_.sum()

            if ll - prev_ll < self.tol * max(abs(prev_ll), 1.0):
                self.converged_ = True
                prev_ll = ll
                break
            prev_ll = ll

        return prev_ll

    # ---------------- public API ----------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarkovSwitchingRegression":
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        yv = np.asarray(y, dtype=float)

        best = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + 977 * i)
            try:
                ll = self._fit_once(Xs, yv, rng)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if best is None or ll > best[0]:
                best = (ll, self.beta_.copy(), self.intercept_.copy(), self.var_.copy(),
                        self.transmat_.copy(), self.startprob_.copy())
        if best is None:
            raise RuntimeError("Markov-switching EM failed from every initialisation")

        (self.loglik_, self.beta_, self.intercept_, self.var_,
         self.transmat_, self.startprob_) = best

        # canonical ordering by residual scale, so state identities are
        # comparable across refits (see src.regime.hmm for the same argument)
        order = np.argsort(self.var_)
        self.beta_ = self.beta_[order]
        self.intercept_ = self.intercept_[order]
        self.var_ = self.var_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]
        self.startprob_ = self.startprob_[order]

        # belief at the end of training, to warm-start the test filter
        log_A = np.log(np.maximum(self.transmat_, 1e-300))
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))
        gamma, _, _ = self._forward_backward(self._log_emission(Xs, yv), log_pi, log_A)
        self.final_belief_ = gamma[-1].copy()
        return self

    def predict(self, X: pd.DataFrame, y_revealed: Optional[np.ndarray] = None,
                horizon: int = 1) -> np.ndarray:
        """Causal out-of-sample prediction.

        ``y_revealed`` is the realised target aligned with ``X``. Because the
        target at time t is not known until t+h, the filter at step t is updated
        only with entries up to ``t - horizon``; the state distribution is then
        propagated the remaining ``horizon`` steps through the transition matrix.
        Passing ``None`` disables updating entirely, which is the fully
        conservative case.
        """
        Xs = self.scaler_.transform(X)
        n = Xs.shape[0]
        K = self.K
        A = self.transmat_
        expert = Xs @ self.beta_.T + self.intercept_[None, :]  # (n, K)

        belief = self.final_belief_.copy()  # posterior at last *observed* target
        n_updated = 0  # number of test targets folded in so far
        out = np.empty(n)

        for t in range(n):
            # fold in every target that has become observable by time t
            while y_revealed is not None and n_updated <= t - horizon:
                j = n_updated
                pred = belief @ A
                r = float(y_revealed[j]) - expert[j]
                lik = np.exp(-0.5 * (r**2 / self.var_) ) / np.sqrt(self.var_)
                post = pred * lik
                s = post.sum()
                belief = post / s if s > 0 and np.isfinite(s) else pred
                n_updated += 1

            # propagate from the last updated index to t
            steps = max(t - n_updated + 1, 1)
            state_dist = belief @ np.linalg.matrix_power(A, steps)
            out[t] = float(np.dot(state_dist, expert[t]))

        return out

    def predict_states(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Filtered state posteriors given observed targets (for diagnostics)."""
        Xs = self.scaler_.transform(X)
        log_A = np.log(np.maximum(self.transmat_, 1e-300))
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))
        gamma, _, _ = self._forward_backward(
            self._log_emission(Xs, np.asarray(y, dtype=float)), log_pi, log_A
        )
        return gamma
