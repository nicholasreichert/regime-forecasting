from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Mode = Literal["hard", "soft"]

DEFAULT_ALPHAS: tuple[float, ...] = tuple(np.logspace(-3, 4, 24))


def _make_ridge(alpha: Optional[float], alphas: Sequence[float], standardize: bool = True) -> Pipeline:
    """Standardized ridge; alpha tuned by GCV on the fitting data when not fixed.

    Standardization is applied inside the pipeline so it is refit on each
    regime's own subsample, and so the penalty is comparable across features
    whose raw scales differ by orders of magnitude. ``standardize=False``
    reproduces the un-scaled variant used in the protocol-ablation study.
    """
    reg = RidgeCV(alphas=tuple(alphas)) if alpha is None else Ridge(alpha=alpha)
    steps = ([("scale", StandardScaler())] if standardize else []) + [("reg", reg)]
    return Pipeline(steps)


@dataclass
class RegimeConditionedRidge:
    """Mixture of per-regime ridge experts, gated by HMM regime probabilities.

    Experts are assigned by the hard argmax of ``train_probs``; regimes with
    fewer than ``min_points_per_regime`` training observations fall back to a
    globally-fitted expert. At prediction time the experts are combined either
    by hard switching (argmax of the gating probabilities) or by soft
    probability weighting.

    ``fit`` and ``predict`` take their gating probabilities separately, which is
    what makes the ablations in :mod:`src.eval.evaluate_regime` possible: the
    expert *partition* and the prediction-time *gate* can be perturbed
    independently.
    """

    alpha: Optional[float] = None  # None => tune by GCV on the training window
    mode: Mode = "soft"
    min_points_per_regime: int = 200
    alphas: tuple[float, ...] = field(default=DEFAULT_ALPHAS)
    standardize: bool = True

    def __post_init__(self):
        self.global_model = _make_ridge(self.alpha, self.alphas, self.standardize)
        self.models: Dict[int, Pipeline] = {}
        self.regimes_trained: List[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, train_probs: np.ndarray) -> "RegimeConditionedRidge":
        if len(X) != len(y) or len(X) != train_probs.shape[0]:
            raise ValueError("X, y, train_probs must have aligned lengths.")

        K = train_probs.shape[1]
        hard = train_probs.argmax(axis=1)

        # global fallback expert, fitted on the pooled training window
        self.global_model = _make_ridge(self.alpha, self.alphas, self.standardize)
        self.global_model.fit(X, y)

        self.models = {}
        self.regimes_trained = []

        for k in range(K):
            idx = np.where(hard == k)[0]
            if len(idx) < self.min_points_per_regime:
                continue
            m = _make_ridge(self.alpha, self.alphas, self.standardize)
            m.fit(X.iloc[idx], y.iloc[idx])
            self.models[k] = m
            self.regimes_trained.append(k)

        self.n_experts_ = len(self.models)
        return self

    def predict(self, X: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
        if len(X) != probs.shape[0]:
            raise ValueError("X and probs must have aligned lengths.")

        K = probs.shape[1]

        expert_preds = np.zeros((len(X), K), dtype=float)
        global_pred: np.ndarray | None = None
        for k in range(K):
            model = self.models.get(k, None)
            if model is None:
                if global_pred is None:
                    global_pred = self.global_model.predict(X)
                expert_preds[:, k] = global_pred
            else:
                expert_preds[:, k] = model.predict(X)

        if self.mode == "hard":
            hard = probs.argmax(axis=1)
            return expert_preds[np.arange(len(X)), hard]

        if self.mode == "soft":
            w = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            return np.sum(expert_preds * w, axis=1)

        raise ValueError(f"Unknown mode: {self.mode}")
