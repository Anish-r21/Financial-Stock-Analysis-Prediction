"""Random Forest regressor wrapper.

Wraps sklearn.ensemble.RandomForestRegressor with a StandardScaler for features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Constants
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RANDOM_STATE = 42


class RandomForestModel:
    """Random Forest regression wrapper with feature scaling.

    Methods:
    - train(X_train, y_train)
    - predict(X_test)
    - get_params(): returns feature importances and model params
    """

    def __init__(self, n_estimators: int = RF_N_ESTIMATORS, max_depth: int = RF_MAX_DEPTH, random_state: int = RANDOM_STATE) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.scaler: StandardScaler = StandardScaler()
        self.model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
        )
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Fit scaler and RandomForest model.

        Parameters
        - X: training features (DataFrame or ndarray)
        - y: target values
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

        Raises RuntimeError if model not trained.
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
        """Return model parameters and feature importances.

        Returns a dict with `feature_importances` mapping feature name to importance.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained to get parameters")

        importances = self.model.feature_importances_
        imp_list = importances.tolist() if hasattr(importances, "tolist") else list(importances)

        if self.feature_names and len(self.feature_names) == len(imp_list):
            feat_map = {name: float(val) for name, val in zip(self.feature_names, imp_list)}
        else:
            feat_map = {f"f{i}": float(val) for i, val in enumerate(imp_list)}

        return {
            "feature_importances": feat_map,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
