from __future__ import annotations

"""Logistic Regression prediction model for player props."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from app.models.base import BasePredictor


class LogisticPredictor(BasePredictor):
    """
    Uses Ridge regression for value prediction and
    Logistic Regression for over/under probability.
    """

    name = "logistic_regression"

    def __init__(self):
        self.regressor = Ridge(alpha=1.0, random_state=42)
        self.classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self._trained_line: float | None = None

    def train(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray | None = None
    ) -> None:
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.regressor.fit(X_scaled, y, sample_weight=sample_weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.regressor.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        predictions = self.predict(X)
        diff = predictions - line
        residual_std = max(np.std(predictions) * 0.5, 1.0)
        proba_over = 1 / (1 + np.exp(-diff / residual_std))
        return proba_over

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "regressor": self.regressor,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.regressor = data["regressor"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]

    def get_feature_importances(self) -> dict[str, float]:
        if not self.feature_names:
            return {}
        coefs = np.abs(self.regressor.coef_)
        total = coefs.sum()
        if total == 0:
            return {}
        normalized = (coefs / total).tolist()
        return dict(zip(self.feature_names, normalized))
