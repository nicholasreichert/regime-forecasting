from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

LossName = Literal["mse", "mae", "qlike"]


@dataclass(frozen=True)
class DMResult:
    stat: float
    p_value: float
    mean_loss_diff: float  # mean(L_a) - mean(L_b); negative => model a better
    n: int
    lag: int
    loss: str

    @property
    def favours(self) -> str:
        if not np.isfinite(self.mean_loss_diff) or self.mean_loss_diff == 0:
            return "tie"
        return "a" if self.mean_loss_diff < 0 else "b"


def _loss_pair(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, loss: LossName
) -> np.ndarray:
    """Loss differential L(e_a) - L(e_b), aligned across both models.

    QLIKE drops observations with a zero realization, and it must drop the same
    ones for both models or the differential compares different samples. That is
    why this works on the pair rather than on each forecast independently.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    if loss == "mse":
        return (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    if loss == "mae":
        return np.abs(y_true - pred_a) - np.abs(y_true - pred_b)
    if loss == "qlike":
        # Patton (2011) robust loss for volatility, evaluated on variances, using
        # the same flooring as the reported metric (see src.eval.metrics).
        from src.eval.metrics import qlike_losses

        la, ok_a = qlike_losses(y_true, pred_a)
        lb, ok_b = qlike_losses(y_true, pred_b)
        keep = ok_a & ok_b
        if keep.sum() == 0:
            return np.empty(0, dtype=float)
        full_a = np.full(len(y_true), np.nan)
        full_b = np.full(len(y_true), np.nan)
        full_a[ok_a] = la
        full_b[ok_b] = lb
        return (full_a - full_b)[keep]
    raise ValueError(f"Unknown loss: {loss}")


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of ``d`` with a Bartlett kernel."""
    n = len(d)
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    total = gamma0
    for k in range(1, lag + 1):
        if k >= n:
            break
        gamma_k = float(np.dot(d[k:], d[:-k]) / n)
        total += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    # A negative estimate is possible in finite samples; fall back to gamma0.
    return total if total > 0 else gamma0


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    loss: LossName = "mse",
    lag: int | None = None,
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy, model a vs. model b.

    H0: E[L(e_a) - L(e_b)] = 0.

    Two corrections matter for this study and are applied here:

    * **HAC variance.** Overlapping h-day targets make the loss differential
      autocorrelated up to lag h-1 by construction, so the naive standard error
      is far too small. A Bartlett/Newey-West estimate is used with lag h-1 by
      default.
    * **Harvey-Leybourne-Newbold (1997) small-sample correction.** The raw DM
      statistic over-rejects at the sample sizes here; the corrected statistic is
      compared against a t distribution with n-1 degrees of freedom.

    A negative statistic favours model a.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must have the same length")

    d = _loss_pair(y_true, pred_a, pred_b, loss)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return DMResult(np.nan, np.nan, float(np.mean(d)) if n else np.nan, n, 0, loss)

    if lag is None:
        lag = max(int(horizon) - 1, 0)

    dbar = float(d.mean())
    lrv = _newey_west_var(d, lag)
    if lrv <= 0:
        return DMResult(np.nan, np.nan, dbar, n, lag, loss)

    dm = dbar / np.sqrt(lrv / n)

    # Harvey-Leybourne-Newbold small-sample adjustment
    h = max(int(horizon), 1)
    adj = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    adj = max(adj, 1e-8)
    dm_star = dm * np.sqrt(adj)

    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_star), df=n - 1)))
    return DMResult(float(dm_star), p, dbar, n, lag, loss)


@dataclass(frozen=True)
class EquivalenceBound:
    """A two-sided confidence interval on RMSE improvement, in percent."""

    point: float  # observed improvement of a over b, %
    lower: float  # most favourable value for b consistent with the data
    upper: float  # most favourable value for a consistent with the data
    n: int
    conf: float

    @property
    def rules_out_improvement_above(self) -> float:
        """Largest improvement of model a over b not excluded by the data."""
        return self.upper


def equivalence_bound(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    conf: float = 0.95,
) -> EquivalenceBound:
    """Bound the RMSE improvement of model a over model b.

    A failure to reject equal accuracy is not evidence of equal accuracy: it is
    also what an underpowered test produces. This converts the Diebold-Mariano
    machinery into the statement a null result actually needs, namely *how large
    an improvement the data can exclude*.

    The interval is built on the mean squared-error differential, using the same
    HAC variance and Harvey-Leybourne-Newbold correction as
    :func:`diebold_mariano`, then mapped monotonically onto the RMSE percentage
    scale via

        improvement(d) = 100 * (1 - sqrt(1 + d / MSE_b)),

    where ``d`` is the mean loss differential. Because the map is monotone
    decreasing in ``d``, the interval endpoints transform directly.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    la = (y_true - pred_a) ** 2
    lb = (y_true - pred_b) ** 2
    d = la - lb
    ok = np.isfinite(d) & np.isfinite(lb)
    d, lb_ok = d[ok], lb[ok]
    n = len(d)

    mse_b = float(np.mean(lb_ok)) if n else np.nan
    if n < 10 or not np.isfinite(mse_b) or mse_b <= 0:
        return EquivalenceBound(np.nan, np.nan, np.nan, n, conf)

    dbar = float(d.mean())
    lag = max(int(horizon) - 1, 0)
    lrv = _newey_west_var(d, lag)
    if lrv <= 0:
        return EquivalenceBound(np.nan, np.nan, np.nan, n, conf)

    h = max(int(horizon), 1)
    adj = max((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n, 1e-8)
    se = np.sqrt(lrv / n) / np.sqrt(adj)  # inflate the SE by the HLN correction
    crit = float(stats.t.ppf(0.5 + conf / 2.0, df=n - 1))

    def to_pct(delta: float) -> float:
        ratio = 1.0 + delta / mse_b
        if ratio <= 0:
            return float("inf")
        return 100.0 * (1.0 - np.sqrt(ratio))

    # improvement is decreasing in d, so the upper CI limit on d gives the lower
    # limit on improvement and vice versa
    return EquivalenceBound(
        point=to_pct(dbar),
        lower=to_pct(dbar + crit * se),
        upper=to_pct(dbar - crit * se),
        n=n,
        conf=conf,
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down correction over a family of tests.

    Returns a name -> reject-H0 mapping. Used because each target is compared
    against several baselines, so uncorrected p-values would overstate
    significance.
    """
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, bool] = {k: False for k in p_values}

    for i, (name, p) in enumerate(items):
        if p <= alpha / (m - i):
            out[name] = True
        else:
            break  # step-down: once we fail, everything after also fails
    return out
