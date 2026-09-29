from pathlib import Path

import pytest

from heart_ai.data import DataValidationError, load_dataset, validate_dataset, validate_features


def test_valid(sample_dataframe): validate_dataset(sample_dataframe)
def test_missing_column(sample_dataframe):
    with pytest.raises(DataValidationError): validate_dataset(sample_dataframe.drop(columns=["Age"]))
def test_invalid_target(sample_dataframe):
    x=sample_dataframe.copy(); x.loc[0,"HeartDisease"]=2
    with pytest.raises(DataValidationError): validate_dataset(x)
def test_missing_file():
    with pytest.raises(FileNotFoundError): load_dataset(Path("does-not-exist.csv"))
def test_empty_features():
    with pytest.raises(ValueError): validate_features([])
def test_unknown_feature():
    with pytest.raises(ValueError): validate_features(["Age","BAD"])
