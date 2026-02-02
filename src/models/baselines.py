from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class BaselineModel:
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
class ZeroReturnBaseline(BaselineModel):
    def __init__(self):
        super().__init__("zero_return")
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.zeros(len(X))
    
class RollingMeanBaseline(BaselineModel):
    def __init__(self):
        super().__init__("rolling_mean")

    def fit(self, X, y):
        self.mean_ = float(y.mean())
        return self
    
    def predict(self, X):
        return np.full(len(X), self.mean_)

class RidgeBaseline(BaselineModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__("ridge")
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    