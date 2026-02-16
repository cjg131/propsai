from __future__ import annotations

"""
Ensemble prediction engine combining all base models.
Uses stacked generalization with performance-weighted fallback.
"""

import signal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.logging_config import get_logger
from app.models.base import BasePredictor
from app.models.bayesian_model import BayesianPredictor
from app.models.logistic_model import LogisticPredictor
from app.models.lstm_model import LSTMPredictor
from app.models.random_forest_model import RandomForestPredictor
from app.models.transformer_model import TransformerPredictor
from app.models.xgboost_model import XGBoostPredictor

logger = get_logger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class EnsembleEngine:
    """
    Ensemble of 6 models with stacked generalization meta-model
    and performance-weighted fallback for cold-start scenarios.
    """

    def __init__(self):
        self.base_models: dict[str, BasePredictor] = {
            "xgboost": XGBoostPredictor(),
            "random_forest": RandomForestPredictor(),
            "logistic_regression": LogisticPredictor(),
            "bayesian": BayesianPredictor(),
        }
        # LSTM and Transformer require sequential per-game data;
        # they are added back when per-game stats are available.
        self.sequential_models: dict[str, BasePredictor] = {
            "lstm": LSTMPredictor(sequence_length=10),
            "transformer": TransformerPredictor(sequence_length=20),
        }
        self.meta_model = Ridge(alpha=1.0)
        self.performance_weights: dict[str, float] = {
            name: 1.0 / len(self.base_models) for name in self.base_models
        }
        self.is_meta_trained = False
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Train all base models and the stacking meta-model.
        Returns training metrics for each model.
        """
        metrics = {}
        base_predictions = {}

        # Train each base model (with 60s timeout per model)
        for name, model in self.base_models.items():
            try:
                logger.info(f"Training {name}...")
                try:
                    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
                    signal.alarm(60)
                    model.train(X, y, sample_weights)
                    signal.alarm(0)
                except TimeoutError:
                    signal.alarm(0)
                    logger.warning(f"{name} training timed out after 60s, skipping")
                    metrics[name] = float("inf")
                    continue

                # Get in-sample predictions for meta-model training
                preds = model.predict(X)
                base_predictions[name] = preds

                # Calculate in-sample RMSE
                rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))
                metrics[name] = rmse
                logger.info(f"{name} trained, RMSE: {rmse:.3f}")
            except Exception as e:
                signal.alarm(0)
                logger.error(f"Error training {name}", error=str(e))
                metrics[name] = float("inf")

        # Train meta-model (stacked generalization)
        valid_predictions = {
            k: v for k, v in base_predictions.items() if k in metrics and metrics[k] < float("inf")
        }

        if len(valid_predictions) >= 2:
            meta_X = pd.DataFrame(valid_predictions)
            try:
                self.meta_model.fit(meta_X, y)
                self.is_meta_trained = True
                meta_preds = self.meta_model.predict(meta_X)
                meta_rmse = float(np.sqrt(np.mean((meta_preds - y.values) ** 2)))
                metrics["meta_ensemble"] = meta_rmse
                logger.info(f"Meta-model trained, RMSE: {meta_rmse:.3f}")
            except Exception as e:
                logger.error("Error training meta-model", error=str(e))

        # Update performance weights (inverse RMSE weighting)
        valid_metrics = {k: v for k, v in metrics.items() if v < float("inf") and k != "meta_ensemble"}
        if valid_metrics:
            total_inv = sum(1.0 / v for v in valid_metrics.values())
            self.performance_weights = {
                k: (1.0 / v) / total_inv for k, v in valid_metrics.items()
            }

        self.is_trained = True
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using stacked generalization or performance-weighted fallback."""
        base_preds = self._get_base_predictions(X)

        if not base_preds:
            return np.zeros(len(X))

        if self.is_meta_trained and len(base_preds) == len(self.base_models):
            # Use stacked generalization
            meta_X = pd.DataFrame(base_preds)
            return self.meta_model.predict(meta_X)
        else:
            # Performance-weighted fallback
            return self._weighted_average(base_preds)

    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Predict probability of going over the line."""
        probas = {}
        for name, model in self.base_models.items():
            try:
                probas[name] = model.predict_proba(X, line)
            except Exception as e:
                logger.warning(f"Error getting proba from {name}", error=str(e))

        if not probas:
            return np.full(len(X), 0.5)

        return self._weighted_average(probas)

    def predict_with_details(
        self, X: pd.DataFrame, line: float
    ) -> dict:
        """
        Full prediction with model contributions, agreement, and ranges.
        Returns detailed prediction data for transparency.
        """
        base_preds = self._get_base_predictions(X)
        base_probas = {}

        for name, model in self.base_models.items():
            try:
                base_probas[name] = model.predict_proba(X, line)
            except Exception:
                pass

        # Ensemble prediction
        if self.is_meta_trained and len(base_preds) == len(self.base_models):
            meta_X = pd.DataFrame(base_preds)
            ensemble_pred = self.meta_model.predict(meta_X)
        else:
            ensemble_pred = self._weighted_average(base_preds)

        ensemble_proba = self._weighted_average(base_probas) if base_probas else np.full(len(X), 0.5)

        # Calculate prediction range from Bayesian model
        bayesian = self.base_models.get("bayesian")
        if isinstance(bayesian, BayesianPredictor):
            try:
                range_low, range_high = bayesian.predict_range(X, confidence=0.9)
            except Exception:
                range_low = ensemble_pred - 3
                range_high = ensemble_pred + 3
        else:
            range_low = ensemble_pred - 3
            range_high = ensemble_pred + 3

        # Calculate ensemble agreement (std of predictions across models)
        if len(base_preds) >= 2:
            all_preds = np.array(list(base_preds.values()))
            agreement = 1.0 - np.clip(np.std(all_preds, axis=0) / (np.mean(np.abs(all_preds), axis=0) + 1e-6), 0, 1)
        else:
            agreement = np.ones(len(X))

        # Model contributions
        contributions = []
        for name in base_preds:
            contributions.append({
                "model_name": name,
                "prediction": float(base_preds[name][-1]) if len(base_preds[name]) > 0 else 0,
                "confidence": float(base_probas.get(name, [0.5])[-1]) if name in base_probas else 0.5,
                "weight": self.performance_weights.get(name, 0),
            })

        # Feature importances (aggregate across models)
        feature_importances = self._aggregate_feature_importances()

        return {
            "predicted_value": ensemble_pred,
            "over_probability": ensemble_proba,
            "under_probability": 1 - ensemble_proba,
            "range_low": range_low,
            "range_high": range_high,
            "agreement": agreement,
            "contributions": contributions,
            "feature_importances": feature_importances,
        }

    def _get_base_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        preds = {}
        for name, model in self.base_models.items():
            try:
                preds[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting with {name}", error=str(e))
        return preds

    def _weighted_average(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute performance-weighted average of predictions."""
        total_weight = 0.0
        weighted_sum = None

        for name, pred in predictions.items():
            weight = self.performance_weights.get(name, 1.0 / len(predictions))
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            total_weight += weight

        if weighted_sum is None or total_weight == 0:
            return np.zeros(1)

        return weighted_sum / total_weight

    def _aggregate_feature_importances(self) -> list[dict]:
        """Aggregate feature importances across all models."""
        all_importances: dict[str, list[float]] = {}

        for name, model in self.base_models.items():
            try:
                importances = model.get_feature_importances()
                for feature, importance in importances.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
            except Exception:
                pass

        # Average importances across models
        result = []
        for feature, values in all_importances.items():
            avg_importance = float(np.mean(values))
            result.append({
                "feature_name": feature,
                "importance": avg_importance,
                "direction": "positive",  # Simplified; would need SHAP for direction
            })

        result.sort(key=lambda x: x["importance"], reverse=True)
        return result[:20]  # Top 20 features

    def save(self, directory: str | None = None) -> None:
        """Save all model artifacts."""
        save_dir = Path(directory) if directory else ARTIFACTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.base_models.items():
            try:
                ext = ".pt" if name in ("lstm", "transformer") else ".joblib"
                model.save(str(save_dir / f"{name}{ext}"))
            except Exception as e:
                logger.error(f"Error saving {name}", error=str(e))

        # Save meta-model and weights
        joblib.dump(
            {
                "meta_model": self.meta_model,
                "performance_weights": self.performance_weights,
                "is_meta_trained": self.is_meta_trained,
            },
            str(save_dir / "meta_model.joblib"),
        )
        logger.info("All models saved", directory=str(save_dir))

    def load(self, directory: str | None = None) -> None:
        """Load all model artifacts."""
        load_dir = Path(directory) if directory else ARTIFACTS_DIR

        for name, model in self.base_models.items():
            try:
                ext = ".pt" if name in ("lstm", "transformer") else ".joblib"
                path = load_dir / f"{name}{ext}"
                if path.exists():
                    model.load(str(path))
                    logger.info(f"Loaded {name}")
            except Exception as e:
                logger.error(f"Error loading {name}", error=str(e))

        # Load meta-model
        meta_path = load_dir / "meta_model.joblib"
        if meta_path.exists():
            try:
                data = joblib.load(str(meta_path))
                self.meta_model = data["meta_model"]
                self.performance_weights = data["performance_weights"]
                self.is_meta_trained = data["is_meta_trained"]
                self.is_trained = True
                logger.info("Meta-model loaded")
            except Exception as e:
                logger.error("Error loading meta-model", error=str(e))


_engine: EnsembleEngine | None = None


def get_ensemble_engine() -> EnsembleEngine:
    global _engine
    if _engine is None:
        _engine = EnsembleEngine()
        # Try to load pre-trained models
        if ARTIFACTS_DIR.exists():
            try:
                _engine.load()
            except Exception:
                pass
    return _engine
