"""Evaluation metrics utilities for regression models.

Provides functions to compute MAE, RMSE, R2, MAPE and compare multiple models.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Compute regression metrics for given true and predicted values.

    Parameters
    - y_true: 1D array of true target values
    - y_pred: 1D array of predicted values
    - model_name: Name of the model for labeling

    Returns
    - dict: { 'model': model_name, 'MAE':..., 'RMSE':..., 'R2':..., 'MAPE':... }
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Avoid divide-by-zero for MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, np.nan, y_true))
        mape = np.nanmean(mape) * 100.0
        mape = float(mape)

    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compare multiple models' metric dictionaries and return a DataFrame sorted by RMSE.

    Parameters
    - results: dict mapping model_name -> metrics dict (as returned by evaluate_model)

    Returns
    - pd.DataFrame: metrics table indexed by model name
    """
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        for k in ("MAE", "RMSE", "R2", "MAPE"):
            row[k] = metrics.get(k)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df[["MAE", "RMSE", "R2", "MAPE"]]
    df = df.sort_values(by="RMSE")
    return df
