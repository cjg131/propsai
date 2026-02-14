from __future__ import annotations
"""LSTM prediction model for sequential game-to-game patterns."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from app.models.base import BasePredictor


class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class LSTMPredictor(BasePredictor):
    name = "lstm"

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model: LSTMNetwork | None = None
        self.feature_names: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is not None and self.std is not None:
            std_safe = np.where(self.std == 0, 1, self.std)
            return (X - self.mean) / std_safe
        return X

    def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Create sequences of length self.sequence_length for LSTM input."""
        sequences = []
        targets = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i : i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])
        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences)

    def train(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray | None = None
    ) -> None:
        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # Normalize
        self.mean = X_np.mean(axis=0)
        self.std = X_np.std(axis=0)
        self.y_mean = float(y_np.mean())
        self.y_std = float(max(y_np.std(), 1e-6))

        X_norm = self._normalize(X_np)
        y_norm = (y_np - self.y_mean) / self.y_std

        if len(X_norm) <= self.sequence_length:
            return

        X_seq, y_seq = self._create_sequences(X_norm, y_norm)

        self.model = LSTMNetwork(input_size=X_np.shape[1])

        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(15):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.y_mean)

        X_np = X.values.astype(np.float32)
        X_norm = self._normalize(X_np)

        if len(X_norm) < self.sequence_length:
            return np.full(len(X), self.y_mean)

        X_seq = self._create_sequences(X_norm)
        X_tensor = torch.FloatTensor(X_seq)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()

        # Denormalize
        predictions = predictions * self.y_std + self.y_mean

        # Pad beginning with mean for sequences we couldn't create
        pad_size = len(X) - len(predictions)
        if pad_size > 0:
            predictions = np.concatenate(
                [np.full(pad_size, self.y_mean), predictions]
            )

        return predictions

    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        predictions = self.predict(X)
        diff = predictions - line
        scale = max(self.y_std * 0.5, 1.0)
        proba_over = 1 / (1 + np.exp(-diff / scale))
        return proba_over

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict() if self.model else None,
                "feature_names": self.feature_names,
                "mean": self.mean,
                "std": self.std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "sequence_length": self.sequence_length,
                "input_size": len(self.feature_names),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        self.feature_names = data["feature_names"]
        self.mean = data["mean"]
        self.std = data["std"]
        self.y_mean = data["y_mean"]
        self.y_std = data["y_std"]
        self.sequence_length = data["sequence_length"]
        if data["model_state"] is not None:
            self.model = LSTMNetwork(input_size=data["input_size"])
            self.model.load_state_dict(data["model_state"])
