from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import _realized_vol_backward, _realized_vol_forward, _rsi
from src.eval.dm_test import diebold_mariano, holm_bonferroni
from src.eval.metrics import mae, qlike, rmse


class TestRealizedVolTargets:
    def test_forward_rv_uses_exactly_the_next_h_returns(self):
        r = pd.Series(np.arange(1.0, 11.0))  # 1..10
        rv = _realized_vol_forward(r, 3)
        # value at t=0 must cover r_1, r_2, r_3 = 2, 3, 4
        expected = np.sqrt(np.mean([4.0, 9.0, 16.0]))
        assert rv.iloc[0] == pytest.approx(expected)

    def test_forward_rv_h1_is_abs_next_return(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(size=200))
        np.testing.assert_allclose(
            _realized_vol_forward(r, 1).dropna().to_numpy(),
            r.abs().shift(-1).dropna().to_numpy(),
            rtol=1e-12,
        )

    def test_forward_rv_has_no_lookahead_beyond_h(self):
        """Changing r_{t+h+1} must not change the target at t."""
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(size=100))
        base = _realized_vol_forward(r, 5)
        perturbed = r.copy()
        perturbed.iloc[50] += 10.0
        after = _realized_vol_forward(perturbed, 5)
        # target at t is unaffected for t such that t+5 < 50, i.e. t <= 44
        np.testing.assert_allclose(base.iloc[:45], after.iloc[:45], rtol=1e-12)
        assert not np.isclose(base.iloc[45], after.iloc[45])

    def test_backward_rv_uses_only_the_past(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(size=100))
        base = _realized_vol_backward(r, 5)
        perturbed = r.copy()
        perturbed.iloc[50] += 10.0
        after = _realized_vol_backward(perturbed, 5)
        np.testing.assert_allclose(base.iloc[:50], after.iloc[:50], rtol=1e-12)

    def test_horizons_are_not_aliases_of_each_other(self):
        """The bug this project originally had: h=1/5/20 with identical marginals."""
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(scale=0.01, size=3000))
        stds = [_realized_vol_forward(r, h).std() for h in (1, 5, 20)]
        # aggregation must reduce dispersion as the window grows
        assert stds[0] > stds[1] > stds[2]


class TestRSI:
    def test_no_nan_when_window_has_no_losses(self):
        """A rolling-mean RSI returns NaN here and punches holes in the panel."""
        r = pd.Series(np.linspace(100, 200, 60))  # monotonically rising
        out = _rsi(r, period=14)
        assert out.iloc[20:].notna().all()
        assert out.iloc[-1] == pytest.approx(100.0)

    def test_bounded(self):
        rng = np.random.default_rng(4)
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
        out = _rsi(px, period=14).dropna()
        assert out.between(0, 100).all()


class TestMetrics:
    def test_qlike_minimised_at_the_truth(self):
        rng = np.random.default_rng(5)
        y = np.abs(rng.normal(0.01, 0.003, 4000))
        perfect = qlike(y, y)
        assert perfect == pytest.approx(0.0, abs=1e-9)
        assert qlike(y, y * 1.3) > perfect
        assert qlike(y, y * 0.7) > perfect

    def test_qlike_penalises_under_prediction_more(self):
        rng = np.random.default_rng(6)
        y = np.abs(rng.normal(0.01, 0.002, 4000))
        assert qlike(y, y * 0.8) > qlike(y, y * 1.2)

    def test_qlike_is_finite_for_zero_forecasts(self):
        y = np.abs(np.random.default_rng(7).normal(0.01, 0.002, 500))
        assert np.isfinite(qlike(y, np.zeros_like(y)))

    def test_rmse_mae_basic(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
        assert mae([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)


class TestDieboldMariano:
    def test_identical_forecasts_give_no_evidence(self):
        rng = np.random.default_rng(8)
        y = rng.normal(size=500)
        p = y + rng.normal(scale=0.1, size=500)
        res = diebold_mariano(y, p, p.copy())
        assert not np.isfinite(res.stat) or abs(res.stat) < 1e-6

    def test_detects_a_clearly_better_model(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=2000)
        good = y + rng.normal(scale=0.1, size=2000)
        bad = y + rng.normal(scale=1.0, size=2000)
        res = diebold_mariano(y, good, bad)
        assert res.stat < 0  # negative favours model a
        assert res.p_value < 0.01
        assert res.favours == "a"

    def test_hac_lag_widens_with_horizon(self):
        """Overlapping targets autocorrelate the loss differential."""
        rng = np.random.default_rng(10)
        y = rng.normal(size=1000)
        a = y + rng.normal(scale=0.5, size=1000)
        b = y + rng.normal(scale=0.55, size=1000)
        assert diebold_mariano(y, a, b, horizon=1).lag == 0
        assert diebold_mariano(y, a, b, horizon=20).lag == 19

    def test_overlapping_errors_reduce_significance(self):
        """With autocorrelated losses the HAC-corrected test must be less certain."""
        rng = np.random.default_rng(11)
        n = 1200
        y = rng.normal(size=n)
        a = y + pd.Series(rng.normal(scale=0.5, size=n)).rolling(20, min_periods=1).mean().to_numpy()
        b = y + pd.Series(rng.normal(scale=0.55, size=n)).rolling(20, min_periods=1).mean().to_numpy()
        naive = diebold_mariano(y, a, b, horizon=1)
        corrected = diebold_mariano(y, a, b, horizon=20)
        assert abs(corrected.stat) < abs(naive.stat)


class TestHolmBonferroni:
    def test_step_down_is_more_conservative_than_uncorrected(self):
        p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
        out = holm_bonferroni(p, alpha=0.05)
        assert out["a"] is True
        assert out["d"] is False
        assert sum(out.values()) < sum(v < 0.05 for v in p.values())

    def test_handles_nan(self):
        out = holm_bonferroni({"a": 0.001, "b": float("nan")}, alpha=0.05)
        assert out["a"] is True
        assert out["b"] is False
