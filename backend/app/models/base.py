from __future__ import annotations
"""Base class for all prediction models in the ensemble."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for all prediction models."""

    name: str = "base"

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray | None = None) -> None:
        """Train the model on historical data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict stat values for given features."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Predict probability of going over the line."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model artifacts to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model artifacts from disk."""
        pass

    def get_feature_importances(self) -> dict[str, float]:
        """Return feature importance scores. Override in subclasses."""
        return {}
