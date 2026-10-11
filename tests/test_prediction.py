import pytest
from src.data_loader import load_data
from src.preprocessing import clean_data, split_features_target
from src.models import make_pipeline
from sklearn.linear_model import LinearRegression


def test_pipeline_predicts_finite_grade_from_real_data():
    df = clean_data(load_data())
    X, y = split_features_target(df)
    pipe = make_pipeline(X, LinearRegression()).fit(X.iloc[:520], y.iloc[:520])
    value = pipe.predict(X.iloc[[520]])[0]
    assert value == pytest.approx(value)
    assert -5 <= value <= 25
