"""Does the regime model pay for a decision, even though it loses on RMSE?

Squared error is not what volatility forecasts are for. They size positions and
set risk limits, and those loss functions are asymmetric and care about different
parts of the distribution. A model with a worse RMSE but a $3.6\\times$ lift on
stress classification could plausibly still be the better input to a risk system,
and if it were, that would be the paper's one positive finding.

Two standard exercises:

*Volatility targeting.* Size a position at ``target / sigma_hat`` and measure how
close realised volatility lands to the target. This is the cleanest decision
metric for a volatility forecast: it has no distributional assumption and the
loss is directly interpretable.

*Value-at-Risk.* Turn the forecast into a quantile using a scaling calibrated on
trailing data only, then test coverage with Kupiec (1995) and Christoffersen
(1998). This probes the tail rather than the centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


@dataclass(frozen=True)
class VolTargetResult:
    realized_ann_vol: float
    target_ann_vol: float
    vol_error: float  # |realized - target|, the quantity being minimised
    sharpe: float
    max_drawdown: float
    mean_leverage: float
    turnover: float
    n: int


def volatility_targeting(
    sigma_hat: pd.Series,
    next_returns: pd.Series,
    target_ann_vol: float = 0.10,
    max_leverage: float = 4.0,
) -> VolTargetResult:
    """Scale exposure inversely to the forecast and see where realised vol lands.

    ``sigma_hat`` is a *daily* volatility forecast made at t; ``next_returns`` is
    the realised return over t+1. Leverage is capped, as any real mandate would
    be, so that a single tiny forecast cannot dominate the result.
    """
    idx = sigma_hat.index.intersection(next_returns.index)
    s = sigma_hat.loc[idx].astype(float)
    r = next_returns.loc[idx].astype(float)
    ok = np.isfinite(s) & np.isfinite(r) & (s > 0)
    s, r = s[ok], r[ok]
    if len(s) < 50:
        return VolTargetResult(*([np.nan] * 7), n=len(s))

    target_daily = target_ann_vol / np.sqrt(TRADING_DAYS)
    w = np.minimum(target_daily / s, max_leverage)
    port = w * r

    realized = float(port.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(port.mean() / port.std(ddof=0) * np.sqrt(TRADING_DAYS)) if port.std() > 0 else np.nan
    curve = (1.0 + port).cumprod()
    dd = float((curve / curve.cummax() - 1.0).min())

    return VolTargetResult(
        realized_ann_vol=realized,
        target_ann_vol=float(target_ann_vol),
        vol_error=float(abs(realized - target_ann_vol)),
        sharpe=sharpe,
        max_drawdown=dd,
        mean_leverage=float(w.mean()),
        turnover=float(np.abs(np.diff(w.to_numpy())).mean()),
        n=int(len(port)),
    )


@dataclass(frozen=True)
class VaRResult:
    alpha: float
    violation_rate: float
    expected_rate: float
    n: int
    n_violations: int
    lr_uc: float
    p_uc: float  # Kupiec unconditional coverage
    lr_ind: float
    p_ind: float  # Christoffersen independence
    lr_cc: float
    p_cc: float  # joint conditional coverage
    mean_var: float


def _kupiec(n: int, x: int, alpha: float) -> tuple[float, float]:
    """Likelihood ratio test that the violation rate equals alpha."""
    if n == 0 or x == 0 or x == n:
        return np.nan, np.nan
    pi = x / n
    ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll_alt = x * np.log(pi) + (n - x) * np.log(1 - pi)
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def _christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """LR test that violations are not clustered (first-order Markov)."""
    h = np.asarray(hits, dtype=int)
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0.0, 1.0) or pi01 in (0.0,) or pi11 in (0.0,):
        return np.nan, np.nan

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
              + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def var_backtest(
    sigma_hat: pd.Series,
    next_returns: pd.Series,
    alpha: float = 0.05,
    calib_window: int = 500,
) -> VaRResult:
    """Backtest a VaR built from the volatility forecast.

    The scaling from forecast to quantile is estimated by filtered historical
    simulation on a trailing window of standardised returns, so it uses only
    information available before the observation being tested. Points before the
    window has filled are dropped rather than calibrated on themselves.
    """
    idx = sigma_hat.index.intersection(next_returns.index)
    s = sigma_hat.loc[idx].astype(float).to_numpy()
    r = next_returns.loc[idx].astype(float).to_numpy()
    ok = np.isfinite(s) & np.isfinite(r) & (s > 0)
    s, r = s[ok], r[ok]
    n_all = len(s)
    if n_all < calib_window + 100:
        return VaRResult(alpha, *([np.nan] * 11))

    z = r / s  # standardised returns
    var_level = np.full(n_all, np.nan)
    for t in range(calib_window, n_all):
        k = np.quantile(z[t - calib_window:t], alpha)  # negative
        var_level[t] = k * s[t]

    valid = np.isfinite(var_level)
    hits = (r[valid] < var_level[valid]).astype(int)
    n, x = int(valid.sum()), int(hits.sum())

    lr_uc, p_uc = _kupiec(n, x, alpha)
    lr_ind, p_ind = _christoffersen_independence(hits)
    if np.isfinite(lr_uc) and np.isfinite(lr_ind):
        lr_cc = lr_uc + lr_ind
        p_cc = float(1 - stats.chi2.cdf(lr_cc, df=2))
    else:
        lr_cc, p_cc = np.nan, np.nan

    return VaRResult(
        alpha=alpha,
        violation_rate=float(x / n) if n else np.nan,
        expected_rate=float(alpha),
        n=n,
        n_violations=x,
        lr_uc=lr_uc, p_uc=p_uc,
        lr_ind=lr_ind, p_ind=p_ind,
        lr_cc=lr_cc, p_cc=p_cc,
        mean_var=float(np.nanmean(var_level[valid])),
    )


def forecast_to_daily_vol(y_pred: pd.Series, horizon: int) -> pd.Series:
    """The RV target is already a daily-scale volatility, so this is a pass-through.

    ``y_rv_h`` is sqrt of the *mean* squared daily return over the horizon, not
    the sum, so it is expressed per day for every h and needs no rescaling. Kept
    as a named function so the assumption is explicit rather than implicit at
    every call site.
    """
    return y_pred.astype(float).clip(lower=1e-8)


def next_period_returns(df: pd.DataFrame, ret_col: str = "ret_1d") -> Optional[pd.Series]:
    """Realised return over t+1, indexed at t."""
    if ret_col not in df.columns:
        return None
    return df[ret_col].shift(-1)
