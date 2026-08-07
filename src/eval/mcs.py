"""Model Confidence Set (Hansen, Lunde and Nason, 2011).

Pairwise Diebold--Mariano tests answer "is A better than B?" one pair at a time,
which is the wrong question when a dozen models are compared at once: the number
of comparisons inflates, and a model can fail to beat any single rival while
still being excluded by the ensemble of them.

The MCS instead returns the set of models that cannot be distinguished from the
best at a given confidence level. It is the natural object for this paper: our
claim is that no regime variant belongs in the set of best models, which is
stronger and cleaner than a list of pairwise failures.

Procedure (the range statistic ``T_R`` variant):

1. Form the loss differentials ``d_ij,t`` for every pair in the current set.
2. Compute ``T_R = max_ij |t_ij|`` where ``t_ij`` studentises the mean
   differential by a bootstrap standard error.
3. Obtain the null distribution of ``T_R`` by stationary block bootstrap, which
   preserves the serial dependence that overlapping targets induce.
4. If the null of equal predictive ability is rejected, eliminate the model with
   the worst standardised average loss and repeat.

The MCS p-value of an eliminated model is the running maximum of the rejection
p-values up to the point of its elimination, so p-values are monotone in
elimination order by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class MCSResult:
    included: List[str]  # models in the confidence set
    p_values: Dict[str, float] = field(default_factory=dict)
    elimination_order: List[str] = field(default_factory=list)
    alpha: float = 0.10
    n_obs: int = 0
    n_boot: int = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"MCSResult(included={self.included}, alpha={self.alpha}, "
                f"n_obs={self.n_obs})")


def _stationary_bootstrap_indices(
    n: int, n_boot: int, block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix, shape (n_boot, n).

    Geometric block lengths with mean ``block_len``. Wrapping keeps every
    resample the same length as the original series, which matters because the
    studentisation below compares statistics across resamples.
    """
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    for t in range(1, n):
        new_block = rng.random(n_boot) < p
        idx[:, t] = np.where(new_block, rng.integers(0, n, size=n_boot), (idx[:, t - 1] + 1) % n)
    return idx


def model_confidence_set(
    losses: Dict[str, np.ndarray],
    alpha: float = 0.10,
    n_boot: int = 2000,
    block_len: float | None = None,
    horizon: int = 1,
    seed: int = 0,
) -> MCSResult:
    """Compute the Model Confidence Set from per-observation loss series.

    ``losses`` maps model name to a loss array; all must be the same length and
    aligned on the same observations. ``block_len`` defaults to the horizon,
    which is the dependence length that overlapping targets induce.
    """
    names = list(losses)
    if len(names) < 2:
        return MCSResult(included=names, p_values={n: 1.0 for n in names}, alpha=alpha)

    L = np.column_stack([np.asarray(losses[n], dtype=float) for n in names])
    ok = np.isfinite(L).all(axis=1)
    L = L[ok]
    n = L.shape[0]
    if n < 20:
        return MCSResult(included=names, p_values={n_: 1.0 for n_ in names}, alpha=alpha, n_obs=n)

    rng = np.random.default_rng(seed)
    bl = float(block_len if block_len is not None else max(horizon, 1))
    boot_idx = _stationary_bootstrap_indices(n, n_boot, bl, rng)

    # Bootstrap resamples of the column means, reused at every elimination round.
    # L[boot_idx] has shape (n_boot, n, n_models); averaging over axis 1 gives the
    # resampled mean loss of each model.
    boot_means = L[boot_idx].mean(axis=1)  # (n_boot, n_models)
    obs_means = L.mean(axis=0)  # (n_models,)

    alive = list(range(len(names)))
    p_values: Dict[str, float] = {}
    elimination: List[str] = []
    running_max = 0.0

    while len(alive) > 1:
        a = np.array(alive)
        om = obs_means[a]
        bm = boot_means[:, a]

        # pairwise mean differentials and their bootstrap standard errors
        d_obs = om[:, None] - om[None, :]  # (m, m)
        d_boot = bm[:, :, None] - bm[:, None, :]  # (n_boot, m, m)
        # centre each bootstrap differential on its observed value
        d_centred = d_boot - d_obs[None, :, :]
        se = d_centred.std(axis=0, ddof=1)
        np.fill_diagonal(se, np.inf)  # ignore self-comparisons
        se = np.where(se > 0, se, np.inf)

        t_obs = np.abs(d_obs) / se
        T_R = float(np.nanmax(t_obs))

        t_boot = np.abs(d_centred) / se[None, :, :]
        T_R_boot = np.nanmax(t_boot, axis=(1, 2))

        p = float(np.mean(T_R_boot >= T_R))
        running_max = max(running_max, p)

        if p > alpha:
            # cannot reject equal predictive ability: everything left survives
            for i in alive:
                p_values.setdefault(names[i], running_max)
            break

        # eliminate the model with the worst standardised average loss
        m = len(alive)
        d_bar = d_obs.sum(axis=1) / max(m - 1, 1)
        se_bar = d_centred.mean(axis=2).std(axis=0, ddof=1)
        se_bar = np.where(se_bar > 0, se_bar, np.inf)
        t_bar = d_bar / se_bar
        worst = int(np.argmax(t_bar))

        eliminated = names[alive[worst]]
        p_values[eliminated] = running_max
        elimination.append(eliminated)
        alive.pop(worst)

    if len(alive) == 1:
        p_values.setdefault(names[alive[0]], max(running_max, 1.0))

    return MCSResult(
        included=[names[i] for i in alive],
        p_values=p_values,
        elimination_order=elimination,
        alpha=alpha,
        n_obs=n,
        n_boot=n_boot,
    )


def squared_error_losses(
    y_true: np.ndarray, preds: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    return {k: (y_true - np.asarray(v, dtype=float)) ** 2 for k, v in preds.items()}


def qlike_losses_for_mcs(
    y_true: np.ndarray, preds: Dict[str, np.ndarray], floor_q: float = 0.01
) -> Dict[str, np.ndarray]:
    """QLIKE loss series aligned on a common mask across all models."""
    from src.eval.metrics import qlike_losses

    y_true = np.asarray(y_true, dtype=float)
    masks = {}
    vals = {}
    for k, v in preds.items():
        li, ok = qlike_losses(y_true, np.asarray(v, dtype=float), floor_q=floor_q)
        masks[k] = ok
        vals[k] = li

    common = np.ones(len(y_true), dtype=bool)
    for ok in masks.values():
        common &= ok

    out = {}
    for k in preds:
        full = np.full(len(y_true), np.nan)
        full[masks[k]] = vals[k]
        out[k] = full[common]
    return out


def mcs_over_models(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    loss: str = "mse",
    alpha: float = 0.10,
    horizon: int = 1,
    n_boot: int = 2000,
    seed: int = 0,
) -> MCSResult:
    if loss == "mse":
        losses = squared_error_losses(y_true, preds)
    elif loss == "qlike":
        losses = qlike_losses_for_mcs(y_true, preds)
    else:
        raise ValueError(f"Unknown loss: {loss}")
    return model_confidence_set(
        losses, alpha=alpha, n_boot=n_boot, horizon=horizon, seed=seed
    )


__all__ = ["MCSResult", "model_confidence_set", "mcs_over_models",
           "squared_error_losses", "qlike_losses_for_mcs"]


def _selftest(seed: int = 0) -> None:  # pragma: no cover - sanity aid
    """Three models, two genuinely tied and one clearly worse."""
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.normal(size=n)
    good_a = y + rng.normal(scale=0.5, size=n)
    good_b = y + rng.normal(scale=0.5, size=n)
    bad = y + rng.normal(scale=1.5, size=n)
    res = mcs_over_models(y, {"good_a": good_a, "good_b": good_b, "bad": bad}, seed=seed)
    print(res)
    print("p-values:", {k: round(v, 3) for k, v in res.p_values.items()})


if __name__ == "__main__":
    _selftest()
