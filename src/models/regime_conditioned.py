from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

Mode = Literal["hard", "soft"]


@dataclass
class RegimeConditionedRidge:
    alpha: float = 1.0
    mode: Mode = "soft"
    min_points_per_regime: int = 200 

    def __post_init__(self):
        self.global_model = Ridge(alpha=self.alpha)
        self.models: Dict[int, Ridge] = {}
        self.regimes_trained: List[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, train_probs: np.ndarray) -> "RegimeConditionedRidge":
        # train global + per-regime expert models
        if len(X) != len(y) or len(X) != train_probs.shape[0]:
            raise ValueError("X, y, train_probs must have aligned lengths.")

        K = train_probs.shape[1]
        hard = train_probs.argmax(axis=1)

        # fit a global model as fallback
        self.global_model.fit(X, y)

        self.models = {}
        self.regimes_trained = []

        for k in range(K):
            idx = np.where(hard == k)[0]
            if len(idx) < self.min_points_per_regime:
                continue
            m = Ridge(alpha=self.alpha)
            m.fit(X.iloc[idx], y.iloc[idx])
            self.models[k] = m
            self.regimes_trained.append(k)

        return self

    def predict(self, X: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
        if len(X) != probs.shape[0]:
            raise ValueError("X and probs must have aligned lengths.")

        K = probs.shape[1]

        # precompute expert predictions
        expert_preds = np.zeros((len(X), K), dtype=float)
        for k in range(K):
            model = self.models.get(k, None)
            if model is None:
                expert_preds[:, k] = self.global_model.predict(X)
            else:
                expert_preds[:, k] = model.predict(X)
            
        if self.mode == "hard":
            hard = probs.argmax(axis=1)
            return expert_preds[np.arange(len(X)), hard]
        
        if self.mode == "soft":
            return np.sum(expert_preds * probs, axis=1)
        
        raise ValueError(f"Unknown mode: {self.mode}")
