from __future__ import annotations

"""Random Forest prediction model for player props."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.models.base import BasePredictor


class RandomForestPredictor(BasePredictor):
    name = "random_forest"

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
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
        # Use individual tree predictions to estimate probability
        tree_predictions = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        )
        # Fraction of trees predicting over the line
        proba_over = np.mean(tree_predictions > line, axis=0)
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
