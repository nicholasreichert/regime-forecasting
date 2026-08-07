"""Tests for the jointly-estimated Markov-switching regression.

The two properties that matter are that it recovers states defined by the
regression relationship (which the two-stage HMM provably does not), and that
its out-of-sample predictor is strictly causal at every horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score

from src.models.markov_switching import MarkovSwitchingRegression, _weighted_ridge


def _switching_data(n=2500, seed=0, p_stay=0.98, noise=0.5):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    beta = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    y = pd.Series((X.to_numpy() * beta[s]).sum(axis=1) + rng.normal(scale=noise, size=n))
    return X, y, s


def _log_space_forward_backward(log_b, log_pi, log_A):
    """Reference implementation the fast scaled version must agree with."""
    T, K = log_b.shape
    la = np.empty((T, K))
    la[0] = log_pi + log_b[0]
    for t in range(1, T):
        la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + log_A, axis=0)
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_A + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = float(logsumexp(la[-1]))
    gamma = np.exp(la + lb - ll)
    xi = np.exp(la[:-1, :, None] + log_A[None] + (log_b[1:] + lb[1:])[:, None, :] - ll).sum(0)
    return gamma, xi, ll


class TestNumericalKernels:
    def test_scaled_forward_backward_matches_log_space(self):
        rng = np.random.default_rng(0)
        T, K = 1500, 3
        log_b = rng.normal(scale=2.0, size=(T, K))
        A = rng.dirichlet(np.ones(K) * 5, K)
        pi = np.full(K, 1.0 / K)

        g1, x1, l1 = _log_space_forward_backward(log_b, np.log(pi), np.log(A))
        g2, x2, l2 = MarkovSwitchingRegression._forward_backward(log_b, np.log(pi), np.log(A))

        np.testing.assert_allclose(g1, g2, atol=1e-10)
        np.testing.assert_allclose(x1, x2, rtol=1e-8)
        assert l1 == pytest.approx(l2, rel=1e-10)

    def test_scaled_recursion_survives_extreme_emissions(self):
        """The scaling exists to stop underflow; check it actually does."""
        rng = np.random.default_rng(1)
        T, K = 800, 3
        log_b = rng.normal(scale=50.0, size=(T, K))  # would underflow unscaled
        A = rng.dirichlet(np.ones(K) * 5, K)
        g, x, ll = MarkovSwitchingRegression._forward_backward(
            log_b, np.log(np.full(K, 1.0 / K)), np.log(A)
        )
        assert np.isfinite(ll)
        assert np.isfinite(g).all()
        np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-8)

    def test_weighted_ridge_matches_sklearn(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(600, 12))
        y = rng.normal(size=600)
        w = rng.random(600)
        for alpha in (0.1, 1.0, 100.0):
            b, c = _weighted_ridge(X, y, w, alpha)
            m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
            np.testing.assert_allclose(b, m.coef_, atol=1e-9)
            assert c == pytest.approx(float(m.intercept_), abs=1e-9)


class TestEstimation:
    def test_recovers_switching_coefficients(self):
        X, y, s = _switching_data()
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=3).fit(X, y)
        slopes = np.sort(m.beta_[:, 0])
        # true slopes are -2 and +2, recoverable only up to state permutation
        assert slopes[0] == pytest.approx(-2.0, abs=0.35)
        assert slopes[1] == pytest.approx(2.0, abs=0.35)

    def test_recovers_the_latent_path(self):
        X, y, s = _switching_data()
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=3).fit(X, y)
        gamma = m.predict_states(X, y.to_numpy())
        assert adjusted_rand_score(s, gamma.argmax(axis=1)) > 0.85

    def test_states_are_canonically_ordered(self):
        X, y, _ = _switching_data()
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=2).fit(X, y)
        assert np.all(np.diff(m.var_) >= 0)

    def test_transition_matrix_is_valid(self):
        X, y, _ = _switching_data()
        m = MarkovSwitchingRegression(K=3, alpha=1.0, seed=0, n_init=2).fit(X, y)
        np.testing.assert_allclose(m.transmat_.sum(axis=1), 1.0, rtol=1e-8)
        assert (m.transmat_ >= 0).all()
        assert m.startprob_.sum() == pytest.approx(1.0)


class TestCausalPrediction:
    @pytest.mark.parametrize("horizon", [1, 5, 20])
    def test_prediction_does_not_use_unrevealed_targets(self, horizon):
        """Corrupting the target at index i must not move predictions before i+h.

        The target at time t is not observed until t+h, so the filter may only
        fold in entries up to t-h. This is the property that makes the
        out-of-sample comparison against the two-stage pipeline fair.
        """
        X, y, _ = _switching_data(n=1600, seed=3)
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=2).fit(
            X.iloc[:1000], y.iloc[:1000]
        )
        Xte = X.iloc[1000:].reset_index(drop=True)
        yte = y.iloc[1000:].to_numpy()

        base = m.predict(Xte, yte, horizon=horizon)
        perturbed = yte.copy()
        perturbed[200] += 50.0
        after = m.predict(Xte, perturbed, horizon=horizon)

        np.testing.assert_allclose(base[: 200 + horizon], after[: 200 + horizon], rtol=1e-10)
        assert not np.allclose(base, after)  # it must use it *eventually*

    def test_no_revealed_targets_is_permitted(self):
        X, y, _ = _switching_data(n=1200, seed=4)
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=2).fit(
            X.iloc[:800], y.iloc[:800]
        )
        pred = m.predict(X.iloc[800:], None, horizon=1)
        assert np.isfinite(pred).all()

    def test_predictions_are_finite_and_in_range(self):
        X, y, _ = _switching_data(n=1400, seed=5)
        m = MarkovSwitchingRegression(K=2, alpha=1.0, seed=0, n_init=2).fit(
            X.iloc[:900], y.iloc[:900]
        )
        pred = m.predict(X.iloc[900:], y.iloc[900:].to_numpy(), horizon=5)
        assert np.isfinite(pred).all()
        assert np.abs(pred).max() < 50 * float(np.abs(y).max())
