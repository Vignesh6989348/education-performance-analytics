import pandas as pd
import pytest
from src.preprocessing import clean_data, split_features_target
from src.evaluate import regression_metrics


def test_clean_removes_duplicates_and_preserves_valid_rows():
    df = pd.DataFrame({"G1":[10,10,12],"G3":[11,11,14]})
    out = clean_data(df)
    assert len(out) == 2
    assert out.G3.tolist() == [11,14]


def test_target_is_separate_and_g1_g2_kept():
    X, y = split_features_target(pd.DataFrame({"G1":[8],"G2":[9],"G3":[10]}))
    assert list(X.columns) == ["G1","G2"]
    assert "G3" not in X
    assert y.iloc[0] == 10


def test_invalid_target_range_rejected():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"G3":[21]}))


def test_metrics_match_known_predictions():
    m = regression_metrics([1,3],[2,3])
    assert m["MAE"] == pytest.approx(.5)
    assert m["RMSE"] == pytest.approx(2**-.5)
    assert m["R2"] == pytest.approx(.5)
