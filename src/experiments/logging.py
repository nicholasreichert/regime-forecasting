from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass    
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

def utc_run_id(prefix: str = "run") -> str:
    # e.g. run_2026-02-02T13-22-10Z
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{ts}"

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def to_jsonable(x: Any) -> Any:
    # convert python objs to something JSON-serializable
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return x

def pack_params(params: Optional[Dict[str, Any]]) -> str:
    # store params as a stable JSOn string
    if not params:
        return "{}"
    clean = {k: to_jsonable(v) for k, v in params.items()}
    return json.dumps(clean, sort_keys=True)

def save_results_csv(rows: list[dict], out_dir: Path, run_id: str) -> Path:
    ensure_dir(out_dir)

    results_df = pd.DataFrame(rows)
    assert isinstance(results_df, pd.DataFrame)

    out_path = out_dir / f"{run_id}.csv"
    results_df.to_csv(out_path, index=False)

    latest_path = out_dir / "latest.csv"
    results_df.to_csv(latest_path, index=False)

    return out_path
