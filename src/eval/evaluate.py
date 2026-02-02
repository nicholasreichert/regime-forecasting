from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.eval.metrics import mae, rmse, directional_accuracy, spearman_corr, top_decile_hit_rate
from src.eval.walk_forward import walk_forward_splits
from src.models.baselines import BaselineModel


def evaluate_model(
        df: pd.DataFrame,
        features: List[str],
        target: str,
        model: BaselineModel,
        train_years: int,
        test_years: int,
        step_years: int,
) -> Dict[str, float]:
    is_vol_target = ("_absret_" in target) or ("_sqret_" in target)

    if is_vol_target:
        metrics = {"rmse": [], "mae": [], "spearman": [], "top_decile_hit": []}
    else:
        metrics = {"rmse": [], "mae": [], "directional_accuracy": []}

    for split in walk_forward_splits(
        df.index, train_years, test_years, step_years
    ):
        train = df.loc[split.train_idx]
        test = df.loc[split.test_idx]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_vol_target:
            metrics["rmse"].append(rmse(y_test, y_pred))
            metrics["mae"].append(mae(y_test, y_pred))
            metrics["spearman"].append(spearman_corr(y_test, y_pred))
            metrics["top_decile_hit"].append(top_decile_hit_rate(y_test, y_pred))
        else:
            metrics["rmse"].append(rmse(y_test, y_pred))
            metrics["mae"].append(mae(y_test, y_pred))
            metrics["directional_accuracy"].append(directional_accuracy(y_test, y_pred))

    return {k: float(sum(v) / len(v)) for k, v in metrics.items()}
 