from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import yaml

TargetType = Literal["ret", "absret", "sqret"]

@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int

@dataclass(frozen=True)
class DataConfig:
    ticker: str
    start: str
    end: Optional[str]
    price_field: str

@dataclass(frozen=True)
class TargetsConfig:
    horizons: List[int]
    target_type: TargetType

@dataclass(frozen=True)
class FeaturesConfig:
    return_lags: int
    rolling_windows: List[int]
    include_volume: bool
    rsi_period: int

@dataclass(frozen=True)
class EvalConfig:
    method: Literal["walk_forward"]
    train_years: int
    test_years: int
    step_years: int

@dataclass(frozen=True)
class HMMConfig:
    K_values: List[int]
    covariance_type: Literal["full", "diag"]
    n_iter: int
    tol: float

@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str
    processed_dir: str
    reports_dir: str
    figures_dir: str

@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    targets: TargetsConfig
    features: FeaturesConfig
    evaluatoin: EvalConfig
    hmm: HMMConfig
    paths: PathsConfig

    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def resolve_dir(self, rel:str) -> Path:
        p = self.root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # normalize YAML nulls / mistakes
    end = raw["data"].get("end", None)
    raw["data"]["end"] = None if end in (None, "null", "") else end

    return Config(
        project = ProjectConfig(**raw["project"]),
        data=DataConfig(**raw["data"]),
        targets=TargetsConfig(**raw["targets"]),
        features=FeaturesConfig(**raw["features"]),
        evaluatoin=EvalConfig(**raw["evaluation"]),
        hmm=HMMConfig(**raw["hmm"]),
        paths=PathsConfig(**raw["paths"]),

    )
