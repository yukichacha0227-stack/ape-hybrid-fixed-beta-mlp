"""Model components for the fixed-beta residual hybrid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class FixedBetaRegressor:
    """Standardized Ridge regression with coefficients recoverable in input units."""

    alpha: float = 1.0

    def fit(self, values: np.ndarray, target: np.ndarray) -> "FixedBetaRegressor":
        self.scaler_ = StandardScaler().fit(values)
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(values), target)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.scaler_.transform(values))

    @property
    def coefficients_original_units_(self) -> np.ndarray:
        return self.model_.coef_ / self.scaler_.scale_

    @property
    def intercept_original_units_(self) -> float:
        return float(
            self.model_.intercept_
            - np.sum(self.model_.coef_ * self.scaler_.mean_ / self.scaler_.scale_)
        )


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
        raise RuntimeError("ResidualMLP requires PyTorch") from exc
    return torch, nn


class ResidualMLP:
    """Factory for the residual-network architecture used in the experiment."""

    def __new__(
        cls,
        input_dim: int,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        dropout: float = 0.05,
    ):
        _, nn = _torch()
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
