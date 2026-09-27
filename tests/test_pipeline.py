import pytest
from heart_ai.models import build_model
from heart_ai.pipeline import train_and_evaluate
@pytest.mark.parametrize("model",["logistic_regression","random_forest","svm"])
def test_models_train(sample_dataframe,model):
    pipe,res=train_and_evaluate(sample_dataframe,model,["Age","Sex","RestingBP","Cholesterol","MaxHR","ExerciseAngina"],.3,42)
    assert pipe is not None and res.model_name==model and 0<=res.metrics["roc_auc"]<=1
def test_bad_model():
    with pytest.raises(ValueError): build_model("bad")
def test_bad_test_size(sample_dataframe):
    with pytest.raises(ValueError): train_and_evaluate(sample_dataframe,"logistic_regression",["Age"],1.0)
def test_cross_validate_model(sample_dataframe):
    from heart_ai.pipeline import cross_validate_model

    results = cross_validate_model(
        df=sample_dataframe,
        model_name="logistic_regression",
        features=[
            "Age",
            "RestingBP",
            "Cholesterol",
            "MaxHR",
        ],
        n_splits=3,
    )

    assert "roc_auc" in results
    assert "mean" in results["roc_auc"]
    assert "std" in results["roc_auc"]

    assert 0 <= results["roc_auc"]["mean"] <= 1
    assert results["roc_auc"]["std"] >= 0
def test_brier_score_is_valid(sample_dataframe):
    _, result = train_and_evaluate(
        sample_dataframe,
        "logistic_regression",
        [
            "Age",
            "Sex",
            "RestingBP",
            "Cholesterol",
            "MaxHR",
            "ExerciseAngina",
        ],
        0.3,
        42,
    )

    brier_score = result.metrics["brier_score"]

    assert 0 <= brier_score <= 1