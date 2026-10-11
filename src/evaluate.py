"""Held-out regression metrics and comparison table."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def compare_predictions(y_test, predictions):
    rows = []
    for name, pred in predictions.items():
        rows.append({"model": name, **regression_metrics(y_test, pred)})
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
