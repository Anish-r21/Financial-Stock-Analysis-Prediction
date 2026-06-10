"""LSTM model for time series prediction using TensorFlow/Keras.

Includes sequence creation, scaling (MinMaxScaler), model building, training,
and prediction utilities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""Note: TensorFlow/Keras imports are deferred to runtime inside methods so
the module can be imported even when TensorFlow is not installed. This allows
the rest of the app to run (non-LSTM models) in environments without TF.
"""

# Constants
LOOKBACK = 60
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int = LOOKBACK) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences of features and aligned targets for LSTM.

    Parameters
    - features: 2D array of shape (n_samples, n_features)
    - target: 1D array of shape (n_samples,)
    - lookback: number of past timesteps to include

    Returns
    - X: 3D array of shape (n_sequences, lookback, n_features)
    - y: 1D array of shape (n_sequences,)
    """
    if len(features) != len(target):
        raise ValueError("Features and target must have the same length")

    Xs = []
    ys = []
    for i in range(lookback, len(features)):
        Xs.append(features[i - lookback:i])
        ys.append(target[i])

    return np.array(Xs), np.array(ys)


class LSTMModel:
    """LSTM regressor that predicts the next day's price from past sequences.

    Usage:
    - Initialize: `model = LSTMModel()`
    - Train: `history = model.train(X, y)` where X is DataFrame/ndarray and y is Series/ndarray
    - Predict: `preds = model.predict(X_test)` returns predictions in original scale
    """

    def __init__(self, lookback: int = LOOKBACK, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size

        self.feature_scaler: MinMaxScaler = MinMaxScaler()
        self.target_scaler: MinMaxScaler = MinMaxScaler()

        self.feature_names: Optional[List[str]] = None
        self.model: Optional[Sequential] = None
        self.is_fitted: bool = False

    def _build_model(self, n_features: int) -> Sequential:
        """Construct the Keras Sequential LSTM model according to the spec."""
        # Local imports to avoid top-level dependency on TensorFlow
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(self.lookback, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer=Adam(), loss="mse")
        return model

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Any:
        """Train the LSTM on provided features and target.

        Parameters
        - X: DataFrame or 2D ndarray of features (ordered by time)
        - y: Series or 1D ndarray of target values aligned with X

        Returns
        - history: Keras History object from `model.fit`
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_vals = X.values
        else:
            X_vals = np.asarray(X)

        y_vals = np.asarray(y).reshape(-1, 1)

        # Scale features and target separately
        X_scaled = self.feature_scaler.fit_transform(X_vals)
        y_scaled = self.target_scaler.fit_transform(y_vals)

        # Create sequences for LSTM
        X_seq, y_seq = create_sequences(X_scaled, y_scaled.ravel(), lookback=self.lookback)

        # Build model
        n_features = X_seq.shape[2]
        self.model = self._build_model(n_features=n_features)

        history = self.model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=VALIDATION_SPLIT,
            verbose=1,
        )

        self.is_fitted = True
        return history

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predictions for provided features.

        Parameters
        - X: DataFrame or 2D ndarray containing the most recent time-ordered features.

        Notes
        - This function will create sequences using the model's `lookback`.
        - Predictions are returned in the original target scale (inverse transformed).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before calling predict()")

        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = np.asarray(X)

        X_scaled = self.feature_scaler.transform(X_vals)
        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), lookback=self.lookback)

        preds_scaled = self.model.predict(X_seq)
        preds = self.target_scaler.inverse_transform(preds_scaled)
        return preds.ravel()

    def get_params(self) -> Dict[str, Any]:
        """Return a dictionary of model parameters and state info."""
        return {
            "lookback": self.lookback,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
