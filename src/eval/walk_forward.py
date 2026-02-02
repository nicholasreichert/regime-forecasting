from __future__ import annotations


from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_idx: pd.Index
    test_idx: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def walk_forward_splits(
        dates: pd.DatetimeIndex,
        train_years: int,
        test_years: int,
        step_years: int,
) -> Iterator[Split]:
    # generate rolling walk-forward splits w/ years
    dates = pd.to_datetime(dates)
    start = dates.min()
    end = dates.max()

    current_train_start = start
    
    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        train_mask = (dates >= current_train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        yield Split(
            train_idx = dates[train_mask],
            test_idx=dates[test_mask],
            train_start=current_train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
        )

        current_train_start += pd.DateOffset(years=step_years)
