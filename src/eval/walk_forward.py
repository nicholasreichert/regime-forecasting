from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_embargoed: int = 0


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    step_years: int,
    embargo: int = 0,
) -> Iterator[Split]:
    """Generate rolling walk-forward splits.

    ``embargo`` drops the final ``embargo`` observations from each training
    window. This is required whenever the target at time t is a function of
    returns over [t+1, t+h]: without it the last h-1 training rows have targets
    that reach into the test period, leaking test-window information into the
    fit. Set ``embargo = h``.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    start = dates.min()
    end = dates.max()

    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    current_train_start = start

    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        train_mask = (dates >= current_train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        train_idx = dates[train_mask]
        test_idx = dates[test_mask]

        # Purge the tail of the training window whose targets overlap the test period.
        n_embargoed = min(int(embargo), len(train_idx))
        if n_embargoed > 0:
            train_idx = train_idx[:-n_embargoed]

        if len(train_idx) > 0 and len(test_idx) > 0:
            yield Split(
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=current_train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                n_embargoed=n_embargoed,
            )

        current_train_start += pd.DateOffset(years=step_years)


def inner_validation_split(
    train_idx: pd.DatetimeIndex,
    val_fraction: float = 0.25,
    embargo: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Carve a trailing validation block out of a training window.

    Used for hyperparameter selection so that no choice (ridge alpha, number of
    regimes, hard vs. soft gating) is ever made using out-of-sample data. The
    same embargo is applied at the inner boundary.
    """
    n = len(train_idx)
    n_val = int(round(n * val_fraction))
    if n_val < 1 or n_val >= n:
        raise ValueError(f"val_fraction={val_fraction} gives an unusable split for n={n}")

    inner_train = train_idx[: n - n_val]
    inner_val = train_idx[n - n_val :]

    n_embargoed = min(int(embargo), len(inner_train))
    if n_embargoed > 0:
        inner_train = inner_train[:-n_embargoed]

    return inner_train, inner_val
