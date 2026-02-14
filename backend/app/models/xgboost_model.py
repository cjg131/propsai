from __future__ import annotations
"""XGBoost prediction model for player props."""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from app.models.base import BasePredictor


class XGBoostPredictor(BasePredictor):
    name = "xgboost"

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def train(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray | None = None
    ) -> None:
        self.feature_names = list(X.columns)
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        predictions = self.predict(X)
        # Estimate probability based on prediction vs line
        # Using a simple sigmoid approximation based on distance from line
        diff = predictions - line
        # Scale factor controls how sharp the probability curve is
        scale = max(np.std(predictions) * 0.5, 1.0)
        proba_over = 1 / (1 + np.exp(-diff / scale))
        return proba_over

    def save(self, path: str) -> None:
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names},
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]

    def get_feature_importances(self) -> dict[str, float]:
        if not self.feature_names:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))
