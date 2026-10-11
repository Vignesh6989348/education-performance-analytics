"""Schema-aware preparation; estimators handle missing values and encoding in pipelines."""
import pandas as pd

TARGET = "G3"
DROP_FROM_FEATURES = {TARGET}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows; preserve all genuine observations otherwise."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is required")
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if cleaned[TARGET].isna().any():
        cleaned = cleaned.dropna(subset=[TARGET]).reset_index(drop=True)
    if not cleaned[TARGET].between(0, 20).all():
        raise ValueError("G3 values must lie on the documented 0–20 scale")
    return cleaned


def split_features_target(df: pd.DataFrame):
    """Return predictors and G3; target is never included in predictors."""
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(float).copy()
    if TARGET in X.columns:
        raise AssertionError("Target leakage detected")
    return X, y
