from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, TargetType

# Standard HAR (Corsi, 2009) lookback windows: daily, weekly, monthly.
HAR_WINDOWS: tuple[int, ...] = (1, 5, 22)


def _log_return(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses Wilder's exponential smoothing (alpha = 1/period), which is what the
    canonical definition calls for. A plain rolling mean is a different
    indicator and, because it produces NaN whenever a window contains no down
    days, silently punched holes in the middle of the feature table.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 => no down moves in the smoothed window => RSI is 100 by
    # definition, not undefined. Keep genuine warm-up NaNs (both legs NaN).
    rsi = rsi.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    return rsi


def _realized_vol_backward(r: pd.Series, window: int) -> pd.Series:
    """Backward-looking realized volatility over the past ``window`` days.

    RV_t^{(w)} = sqrt( mean_{i=0..w-1} r_{t-i}^2 ). Uses information through t
    only, so it is a valid feature at time t.
    """
    return np.sqrt((r**2).rolling(window, min_periods=window).mean())


def _realized_vol_forward(r: pd.Series, horizon: int) -> pd.Series:
    """Realized volatility over the *future* window [t+1, t+h].

    RV_{t+1:t+h} = sqrt( (1/h) * sum_{i=1..h} r_{t+i}^2 ).

    This is the standard multi-horizon volatility forecasting target. Note that
    for h = 1 it reduces exactly to |r_{t+1}|, so it nests the single-day
    absolute-return target.
    """
    # Backward rolling mean of squared returns, then shift back by h so that the
    # value at index t covers [t+1, t+h].
    fwd_ms = (r**2).rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    return np.sqrt(fwd_ms)


def build_feature_table(cfg: Config, raw: pd.DataFrame) -> pd.DataFrame:
    """Build a single leakage-free table of features at t and targets at t+h.

    Two families of targets are emitted:

    ``y_{target_type}_h{h}``
        Point-in-time transform of the single return h days ahead, e.g.
        ``y_absret_h5 = |r_{t+5}|``. Retained for backwards compatibility; note
        that these are the *same* series at different leads, so the marginal
        distribution does not vary with h.

    ``y_rv_h{h}``
        Realized volatility over the whole window [t+1, t+h]. This is the
        headline target: it is a genuinely different problem for each h and is
        the quantity the HAR/GARCH literature forecasts.
    """
    price_field = cfg.data.price_field
    if price_field not in raw.columns:
        raise ValueError(f"price_field '{price_field}' not in raw columns={list(raw.columns)}")

    df = raw.copy()

    adj_close = df[price_field].astype(float)
    volume = df["Volume"].astype(float)

    # daily log return r_t (uses t and t-1)
    r = _log_return(adj_close).rename("ret_1d")

    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = r

    # lagged returns: ret_lag_1 ... ret_lag_N
    L = int(cfg.features.return_lags)
    for lag in range(1, L + 1):
        out[f"ret_lag_{lag}"] = out["ret_1d"].shift(lag)

    # rolling windows
    for w in cfg.features.rolling_windows:
        w = int(w)
        out[f"ret_mean_{w}"] = out["ret_1d"].rolling(w, min_periods=w).mean()
        out[f"ret_vol_{w}"] = out["ret_1d"].rolling(w, min_periods=w).std(ddof=0)

    # HAR components: backward realized volatility at daily/weekly/monthly scales.
    # These are the regressors of the standard HAR-RV model and are also
    # legitimate predictors for every other model in the comparison.
    for w in HAR_WINDOWS:
        out[f"rv_bwd_{w}"] = _realized_vol_backward(out["ret_1d"], int(w))

    # volume-based features
    if cfg.features.include_volume:
        out["vol_chg_1d"] = np.log(volume).diff()
        for w in cfg.features.rolling_windows:
            w = int(w)
            v_mean = volume.rolling(w, min_periods=w).mean()
            v_std = volume.rolling(w, min_periods=w).std(ddof=0)
            out[f"vol_z_{w}"] = (volume - v_mean) / v_std

    out[f"rsi_{cfg.features.rsi_period}"] = _rsi(adj_close, period=int(cfg.features.rsi_period))

    # ---- targets -------------------------------------------------------
    target_type: TargetType = cfg.targets.target_type
    for h in cfg.targets.horizons:
        h = int(h)

        # legacy point-in-time target: transform of r_{t+h}
        if target_type == "ret":
            y = out["ret_1d"].shift(-h)
        elif target_type == "absret":
            y = out["ret_1d"].abs().shift(-h)
        elif target_type == "sqret":
            y = (out["ret_1d"] ** 2).shift(-h)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        out[f"y_{target_type}_h{h}"] = y

        # headline target: realized volatility over [t+1, t+h]
        out[f"y_rv_h{h}"] = _realized_vol_forward(out["ret_1d"], h)

    out = out.dropna().copy()

    # The HMM treats consecutive rows as consecutive time steps, so the feature
    # table must not have holes punched in the middle of its date range.
    pos = raw.index.get_indexer(out.index)
    if len(pos) > 1 and (np.diff(pos) != 1).any():
        n_gaps = int((np.diff(pos) != 1).sum())
        raise ValueError(
            f"Feature table has {n_gaps} interior gap(s) relative to the trading "
            "calendar. Consecutive-observation models (HMM, HAR) assume contiguity."
        )

    return out
