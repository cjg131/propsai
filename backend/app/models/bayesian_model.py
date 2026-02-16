from __future__ import annotations

"""Bayesian prediction model using PyMC for uncertainty quantification."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from app.models.base import BasePredictor


class BayesianPredictor(BasePredictor):
    """
    Bayesian model that naturally outputs probability distributions.
    Uses BayesianRidge as a fast approximation. For full PyMC inference,
    see the train_full_bayesian method (slower but more accurate).
    """

    name = "bayesian"

    def __init__(self):
        self.model = BayesianRidge(
            max_iter=500,
            tol=1e-4,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True,
        )
        self.feature_names: list[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is not None and self.std_ is not None:
            std_safe = np.where(self.std_ == 0, 1, self.std_)
            return (X - self.mean_) / std_safe
        return X

    def train(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray | None = None
    ) -> None:
        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float64)
        self.mean_ = X_np.mean(axis=0)
        self.std_ = X_np.std(axis=0)
        X_norm = self._normalize(X_np)
        self.model.fit(X_norm, y.values, sample_weight=sample_weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.values.astype(np.float64)
        X_norm = self._normalize(X_np)
        return self.model.predict(X_norm)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates (mean and std)."""
        X_np = X.values.astype(np.float64)
        X_norm = self._normalize(X_np)
        mean, std = self.model.predict(X_norm, return_std=True)
        return mean, std

    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """
        Predict probability of going over the line using the
        posterior predictive distribution.
        """
        from scipy import stats

        mean, std = self.predict_with_uncertainty(X)
        # P(value > line) = 1 - CDF(line)
        proba_over = 1 - stats.norm.cdf(line, loc=mean, scale=std)
        return proba_over

    def predict_range(
        self, X: pd.DataFrame, confidence: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict confidence interval for the stat value."""
        from scipy import stats

        mean, std = self.predict_with_uncertainty(X)
        alpha = (1 - confidence) / 2
        z = stats.norm.ppf(1 - alpha)
        lower = mean - z * std
        upper = mean + z * std
        return lower, upper

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "mean_": self.mean_,
                "std_": self.std_,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.mean_ = data["mean_"]
        self.std_ = data["std_"]

    def get_feature_importances(self) -> dict[str, float]:
        if not self.feature_names:
            return {}
        coefs = np.abs(self.model.coef_)
        total = coefs.sum()
        if total == 0:
            return {}
        normalized = (coefs / total).tolist()
        return dict(zip(self.feature_names, normalized))
