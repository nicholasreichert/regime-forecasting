from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import Config


@dataclass(frozen=True)
class RawMarketData:
    df: pd.DataFrame
    raw_path: Path

def _raw_filename(cfg: Config) -> str:
    return f"{cfg.data.ticker}_{cfg.data.start}_raw.parquet"

def load_or_download_equity_data(cfg:Config, force:bool = False) -> RawMarketData:
    # downloads SPY OHLCV with yfinance and caches it into data/raw as Parquet
    raw_dir = cfg.resolve_dir(cfg.paths.raw_dir)
    raw_path = raw_dir / _raw_filename(cfg)

    if raw_path.exists() and not force:
        df = pd.read_parquet(raw_path)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return RawMarketData(df=df, raw_path = raw_path)

    ticker = cfg.data.ticker
    start = cfg.data.start
    end: Optional[str] = cfg.data.end

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="column",
    )  

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker={ticker}. Check ticker/date range.")
    
    # normalize index/columns
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    #df.columns = [str(c) for c in df.columns]

    # check for multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c) for c in df.columns]


    # enforce req fields
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing columns from yfinance: {missing}. Got columns={list(df.columns)}")

    # cache
    df.to_parquet(raw_path)

    return RawMarketData(df=df, raw_path=raw_path)