"""Linear regression model wrapper.

Provides a scikit-learn LinearRegression wrapped with a StandardScaler and
convenience methods for training, prediction, and parameter inspection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Constants
RANDOM_STATE = 42


class LinearRegressionModel:
    """Linear Regression wrapper with internal feature scaling.

    Methods
    - train(X_train, y_train): fit scaler and linear model
    - predict(X): return numpy array of predictions
    - get_params(): returns model coefficients and intercept
    """

    def __init__(self) -> None:
        """Initialize scaler and linear regression model."""
        self.scaler: StandardScaler = StandardScaler()
        self.model: LinearRegression = LinearRegression()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Fit the scaler and linear regression model.

        Parameters
        - X: training features (DataFrame or 2D ndarray)
        - y: target values (Series or 1D ndarray)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_vals = X.values
        else:
            X_vals = np.asarray(X)

        y_vals = np.asarray(y).ravel()

        X_scaled = self.scaler.fit_transform(X_vals)
        self.model.fit(X_scaled, y_vals)
        self.is_fitted = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale features and return predictions.

        Parameters
        - X: features to predict on (DataFrame or 2D ndarray)

        Returns
        - numpy.ndarray: 1D array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calling predict()")

        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = np.asarray(X)

        X_scaled = self.scaler.transform(X_vals)
        preds = self.model.predict(X_scaled)
        return np.asarray(preds).ravel()

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters and feature importances (coefficients).

        Returns a dictionary with keys: `coefficients`, `intercept`, `feature_names`.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained to get parameters")

        coef = self.model.coef_
        intercept = float(self.model.intercept_)

        coef_list = coef.tolist() if hasattr(coef, "tolist") else list(coef)

        feature_map: Dict[str, float]
        if self.feature_names and len(self.feature_names) == len(coef_list):
            feature_map = {name: float(val) for name, val in zip(self.feature_names, coef_list)}
        else:
            feature_map = {f"f{i}": float(val) for i, val in enumerate(coef_list)}

        return {
            "coefficients": feature_map,
            "intercept": intercept,
        }
