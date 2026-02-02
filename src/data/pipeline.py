from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.features import build_feature_table
from src.data.make_dataset import load_or_download_equity_data


@dataclass(frozen=True)
class ProcessedDataset:
    df: pd.DataFrame
    processed_path: Path

def _processed_filename(cfg: Config) -> str:
    horizons = "-".join(str(h) for h in cfg.targets.horizons)
    return f"{cfg.data.ticker}_{cfg.targets.target_type}_h{horizons}_features.parquet"

def build_and_save_processed_dataset(cfg:Config, force_raw:bool = False) ->ProcessedDataset:
    processed_dir = cfg.resolve_dir(cfg.paths.processed_dir)
    processed_path = processed_dir / _processed_filename(cfg)

    raw = load_or_download_equity_data(cfg, force=force_raw)
    feats = build_feature_table(cfg, raw.df)

    feats.to_parquet(processed_path)

    return ProcessedDataset(df=feats, processed_path=processed_path)
