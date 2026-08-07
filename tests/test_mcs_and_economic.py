"""Tests for the newer statistical machinery: MCS, equivalence bounds, risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.dm_test import equivalence_bound
from src.eval.economic import (
    _christoffersen_independence,
    _kupiec,
    var_backtest,
    volatility_targeting,
)
from src.eval.mcs import mcs_over_models, model_confidence_set


class TestModelConfidenceSet:
    def test_eliminates_a_clearly_worse_model(self):
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.normal(size=n)
        preds = {
            "good_a": y + rng.normal(scale=0.5, size=n),
            "good_b": y + rng.normal(scale=0.5, size=n),
            "bad": y + rng.normal(scale=1.5, size=n),
        }
        res = mcs_over_models(y, preds, n_boot=500, seed=0)
        assert "bad" not in res.included
        assert set(res.included) == {"good_a", "good_b"}
        assert res.p_values["bad"] < 0.10

    def test_keeps_everything_when_all_models_tie(self):
        rng = np.random.default_rng(1)
        n = 1500
        y = rng.normal(size=n)
        preds = {f"m{i}": y + rng.normal(scale=0.5, size=n) for i in range(4)}
        res = mcs_over_models(y, preds, n_boot=500, seed=1)
        assert len(res.included) == 4

    def test_single_model_is_trivially_the_set(self):
        res = model_confidence_set({"only": np.ones(100)})
        assert res.included == ["only"]

    def test_p_values_are_monotone_in_elimination_order(self):
        """MCS p-values are a running maximum, so eliminations cannot un-reject."""
        rng = np.random.default_rng(2)
        n = 2000
        y = rng.normal(size=n)
        preds = {
            "best": y + rng.normal(scale=0.3, size=n),
            "mid": y + rng.normal(scale=0.9, size=n),
            "worst": y + rng.normal(scale=2.5, size=n),
        }
        res = mcs_over_models(y, preds, n_boot=500, seed=2)
        ordered = [res.p_values[m] for m in res.elimination_order]
        assert ordered == sorted(ordered)

    def test_block_length_tracks_horizon(self):
        """Longer blocks must be usable without error and should not crash on h>1."""
        rng = np.random.default_rng(3)
        n = 800
        y = rng.normal(size=n)
        preds = {"a": y + rng.normal(scale=0.5, size=n), "b": y + rng.normal(scale=0.6, size=n)}
        res = mcs_over_models(y, preds, horizon=20, n_boot=300, seed=3)
        assert res.n_obs == n


class TestEquivalenceBound:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(4)
        n = 3000
        y = rng.normal(size=n)
        a = y + rng.normal(scale=0.5, size=n)
        b = y + rng.normal(scale=0.5, size=n)
        eb = equivalence_bound(y, a, b)
        assert eb.lower <= eb.point <= eb.upper

    def test_identical_models_centre_on_zero(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=1000)
        p = y + rng.normal(scale=0.4, size=1000)
        eb = equivalence_bound(y, p, p.copy())
        assert eb.point == pytest.approx(0.0, abs=1e-9)
        assert eb.lower <= 0 <= eb.upper

    def test_clearly_better_model_has_positive_lower_bound(self):
        rng = np.random.default_rng(6)
        n = 4000
        y = rng.normal(size=n)
        good = y + rng.normal(scale=0.2, size=n)
        bad = y + rng.normal(scale=1.0, size=n)
        eb = equivalence_bound(y, good, bad)
        assert eb.point > 0
        assert eb.lower > 0  # excludes "no improvement"

    def test_bound_tightens_with_sample_size(self):
        rng = np.random.default_rng(7)
        widths = []
        for n in (400, 8000):
            y = rng.normal(size=n)
            a = y + rng.normal(scale=0.5, size=n)
            b = y + rng.normal(scale=0.5, size=n)
            eb = equivalence_bound(y, a, b)
            widths.append(eb.upper - eb.lower)
        assert widths[1] < widths[0]


class TestVolatilityTargeting:
    def test_perfect_forecast_hits_the_target(self):
        rng = np.random.default_rng(8)
        n = 4000
        idx = pd.bdate_range("2010-01-01", periods=n)
        sigma = pd.Series(np.full(n, 0.01), index=idx)
        r = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
        res = volatility_targeting(sigma, r.shift(-1), target_ann_vol=0.10)
        assert res.realized_ann_vol == pytest.approx(0.10, abs=0.012)

    def test_noisier_forecast_has_larger_error(self):
        rng = np.random.default_rng(9)
        n = 4000
        idx = pd.bdate_range("2010-01-01", periods=n)
        true_s = pd.Series(
            np.clip(0.01 * np.exp(0.3 * np.cumsum(rng.normal(scale=0.05, size=n))), 0.002, 0.05),
            index=idx,
        )
        r = pd.Series(rng.normal(scale=true_s.to_numpy()), index=idx)
        nxt = r.shift(-1)
        good = volatility_targeting(true_s, nxt)
        noisy = volatility_targeting(
            true_s * np.exp(rng.normal(scale=0.6, size=n)), nxt
        )
        assert good.vol_error < noisy.vol_error


class TestVaRTests:
    def test_kupiec_accepts_correct_coverage(self):
        lr, p = _kupiec(n=1000, x=50, alpha=0.05)
        assert p > 0.5

    def test_kupiec_rejects_wrong_coverage(self):
        lr, p = _kupiec(n=1000, x=150, alpha=0.05)
        assert p < 0.01

    def test_independence_rejects_clustered_violations(self):
        # 40 violations, all consecutive: maximally clustered
        hits = np.zeros(1000, dtype=int)
        hits[200:240] = 1
        lr, p = _christoffersen_independence(hits)
        assert p < 0.01

    def test_independence_accepts_scattered_violations(self):
        rng = np.random.default_rng(10)
        hits = (rng.random(2000) < 0.05).astype(int)
        lr, p = _christoffersen_independence(hits)
        assert p > 0.05

    def test_var_backtest_reaches_nominal_coverage(self):
        rng = np.random.default_rng(11)
        n = 3000
        idx = pd.bdate_range("2010-01-01", periods=n)
        sigma = pd.Series(np.full(n, 0.01), index=idx)
        r = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
        res = var_backtest(sigma, r.shift(-1), alpha=0.05, calib_window=500)
        assert res.violation_rate == pytest.approx(0.05, abs=0.02)
        assert res.p_uc > 0.05
