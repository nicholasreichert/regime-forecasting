"""The claims this paper rests on are causality claims, so they get tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.walk_forward import inner_validation_split, walk_forward_splits
from src.regime.hmm import fit_hmm_and_infer_probs, forward_filter_probs


def _synthetic_market(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Two-state volatility-switching series with an obvious regime structure."""
    rng = np.random.default_rng(seed)
    state = 0
    rets = []
    for _ in range(n):
        # persistent switching
        if rng.random() < (0.01 if state == 0 else 0.05):
            state = 1 - state
        rets.append(rng.normal(0, 0.005 if state == 0 else 0.02))
    r = np.array(rets)
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame({"ret_1d": r}, index=idx)
    df["ret_vol_20"] = df["ret_1d"].rolling(20, min_periods=20).std(ddof=0)
    return df.dropna()


class TestForwardFilterIsCausal:
    def test_filtered_prob_ignores_the_future(self):
        """gamma_t must not change when observations after t are deleted.

        This is the single property that separates a filtered posterior from a
        smoothed one, and the mistake that most often makes regime pipelines
        look better than they are.
        """
        rng = np.random.default_rng(0)
        T, K = 200, 3
        log_emission = rng.normal(size=(T, K))
        startprob = np.full(K, 1.0 / K)
        transmat = np.full((K, K), 1.0 / K)

        full, _ = forward_filter_probs(None, startprob, transmat, log_emission)

        for cut in (50, 120, 199):
            truncated, _ = forward_filter_probs(
                None, startprob, transmat, log_emission[: cut + 1]
            )
            np.testing.assert_allclose(full[: cut + 1], truncated, rtol=1e-10, atol=1e-12)

    def test_probabilities_are_normalised(self):
        rng = np.random.default_rng(1)
        probs, last = forward_filter_probs(
            None,
            np.array([0.5, 0.5]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            rng.normal(size=(300, 2)),
        )
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-10)
        np.testing.assert_allclose(last, probs[-1], rtol=1e-12)

    def test_hmm_test_probs_do_not_depend_on_later_test_data(self):
        """Truncating the test window must not change earlier test-window beliefs."""
        df = _synthetic_market()
        train, test = df.iloc[:800], df.iloc[800:]

        full = fit_hmm_and_infer_probs(train, test, K=2, seed=0, tol=1e-3, n_iter=50)
        short = fit_hmm_and_infer_probs(train, test.iloc[:100], K=2, seed=0, tol=1e-3, n_iter=50)

        np.testing.assert_allclose(
            full.test_probs[:100], short.test_probs, rtol=1e-8, atol=1e-10
        )


class TestCanonicalStateOrdering:
    def test_states_sorted_by_volatility(self):
        df = _synthetic_market()
        res = fit_hmm_and_infer_probs(df.iloc[:800], df.iloc[800:], K=3, seed=0, n_iter=100)
        vol_means = res.model.means_[:, 2]  # emission dim 2 is realized vol
        assert np.all(np.diff(vol_means) > 0), (
            "states must be relabelled low->high volatility so identities are "
            "comparable across walk-forward folds"
        )

    def test_relabelling_preserves_model_validity(self):
        df = _synthetic_market()
        res = fit_hmm_and_infer_probs(df.iloc[:800], df.iloc[800:], K=3, seed=0, n_iter=100)
        np.testing.assert_allclose(res.model.transmat_.sum(axis=1), 1.0, rtol=1e-8)
        np.testing.assert_allclose(res.model.startprob_.sum(), 1.0, rtol=1e-8)
        np.testing.assert_allclose(res.train_probs.sum(axis=1), 1.0, rtol=1e-8)


class TestWalkForwardSplits:
    def test_train_and_test_never_overlap(self):
        dates = pd.bdate_range("2005-01-01", "2026-01-01")
        for s in walk_forward_splits(dates, 6, 1, 1, embargo=0):
            assert len(s.train_idx.intersection(s.test_idx)) == 0
            assert s.train_idx.max() < s.test_idx.min()

    @pytest.mark.parametrize("embargo", [1, 5, 20])
    def test_embargo_removes_exactly_h_rows(self, embargo):
        dates = pd.bdate_range("2005-01-01", "2026-01-01")
        plain = list(walk_forward_splits(dates, 6, 1, 1, embargo=0))
        purged = list(walk_forward_splits(dates, 6, 1, 1, embargo=embargo))
        assert len(plain) == len(purged)
        for a, b in zip(plain, purged):
            assert len(a.train_idx) - len(b.train_idx) == embargo
            assert b.test_idx.equals(a.test_idx)

    def test_embargo_gap_exceeds_horizon(self):
        """After purging, no training target window can reach the test fold."""
        dates = pd.bdate_range("2005-01-01", "2026-01-01")
        h = 20
        for s in walk_forward_splits(dates, 6, 1, 1, embargo=h):
            last_train = dates.get_loc(s.train_idx.max())
            first_test = dates.get_loc(s.test_idx.min())
            assert first_test - last_train > h - 1

    def test_inner_split_is_ordered_and_embargoed(self):
        dates = pd.bdate_range("2005-01-01", "2011-01-01")
        inner_tr, inner_val = inner_validation_split(dates, val_fraction=0.25, embargo=5)
        assert inner_tr.max() < inner_val.min()
        assert len(inner_tr.intersection(inner_val)) == 0
        no_embargo, _ = inner_validation_split(dates, val_fraction=0.25, embargo=0)
        assert len(no_embargo) - len(inner_tr) == 5
