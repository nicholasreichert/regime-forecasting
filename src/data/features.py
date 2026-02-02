from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, TargetType


def _log_return(price: pd.Series) -> pd.Series:
    return np.log(price).diff()

def _rsi(close: pd.Series, period: int=14) -> pd.Series:
    # rsi implementation using Wilder's EMA approach
    delta = close.diff()
    gain = delta.clip(lower = 0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def build_feature_table(cfg: Config, raw: pd.DataFrame) -> pd.DataFrame:
    # builds a single table with
    # features at time t
    # targets at time t+h
    # returns dataframe indexed by date w/ no leakage
    price_field = cfg.data.price_field
    if price_field not in raw.columns:
        raise ValueError(f"price_field]'{price_field}' not in raw columns={list(raw.columns)}")

    df = raw.copy()

    adj_close = df[price_field].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # daily log return r_t (uses t and t-1)
    r = _log_return(adj_close).rename("ret_1d")

    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = r

    # lagged returns: ret_1d_lag 1 ... lagN
    L = int(cfg.features.return_lags)
    for lag in range(1, L+1):
        out[f"ret_lag_{lag}"] = out["ret_1d"].shift(lag)
    
    # rolling windows
    for w in cfg.features.rolling_windows:
        w = int(w)
        out[f"ret_mean_{w}"] = out["ret_1d"].rolling(w, min_periods = w).mean()
        out[f"ret_vol_{w}"] = out["ret_1d"].rolling(w, min_periods=w).std(ddof=0)
    
    # volume-based features
    if cfg.features.include_volume:
        out["vol_chg_1d"] = np.log(volume).diff()
        for w in cfg.features.rolling_windows:
            w = int(w)
            v_mean = volume.rolling(w, min_periods=w).mean()
            v_std = volume.rolling(w, min_periods=w).std(ddof=0)
            out[f"vol_z_{w}"] = (volume - v_mean) / v_std
    
    out[f"rsi_{cfg.features.rsi_period}"] = _rsi(close, period=int(cfg.features.rsi_period))

    # targets
    # define target on returns, then shift by horizon
    # y(t) = r(t+h)

    target_type: TargetType = cfg.targets.target_type
    for h in cfg.targets.horizons:
        h = int(h)
        if target_type == "ret":
            y = out["ret_1d"].shift(-h)
        elif target_type == "absret":
            y = out["ret_1d"].abs().shift(-h)
        elif target_type == "sqret":
            y = (out["ret_1d"] ** 2).shift(-h)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        out[f"y_{target_type}_h{h}"] = y

    out = out.dropna().copy()

    return out
