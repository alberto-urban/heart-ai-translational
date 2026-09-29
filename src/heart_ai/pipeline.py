from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TARGET, TEST_SIZE
from .data import validate_dataset, validate_features
from .models import build_model

a=dataclass
@dataclass(frozen=True)
class AnalysisResult:
    model_name:str; features:list[str]; metrics:dict; feature_importance:dict[str,float]
def build_pipeline(model_name:str,features:list[str])->Pipeline:
    from .preprocessing import build_preprocessor
    validate_features(features)
    return Pipeline([("preprocessor",build_preprocessor(features)),("classifier",build_model(model_name))])
def cross_validate_model(
    df: pd.DataFrame,
    model_name: str,
    features: list[str],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Evaluate a model using stratified k-fold cross-validation."""
    validate_dataset(df)
    validate_features(features)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    X = df[features]
    y = df[TARGET]

    pipeline = build_pipeline(
        model_name=model_name,
        features=features,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "brier_score": "neg_brier_score",
    }

    scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results = {}

    for metric_name in scoring:
        values = scores[f"test_{metric_name}"]

        if metric_name == "brier_score":
            values = -values

        results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return results
def evaluate(y_true,y_pred,y_score)->dict:
    fpr,tpr,thr=roc_curve(y_true,y_score)
    probabilities, observed_frequencies = calibration_curve(y_true,y_score,n_bins=10,strategy="uniform")
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "precision":float(precision_score(y_true,y_pred,zero_division=0)),
        "recall":float(recall_score(y_true,y_pred,zero_division=0)),
        "f1":float(f1_score(y_true,y_pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y_true,y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "confusion_matrix":confusion_matrix(y_true,y_pred).tolist(),
        "roc_curve":{
            "fpr":fpr.tolist(),
            "tpr":tpr.tolist(),
            "thresholds":thr.tolist()
        },
        "calibration_curve": {
            "probabilities": probabilities.tolist(),
            "observed_frequencies": observed_frequencies.tolist(),
        },
    }
def importance(pipe:Pipeline)->dict[str,float]:
    p=pipe.named_steps["preprocessor"]; c=pipe.named_steps["classifier"]; names=list(p.get_feature_names_out())
    if hasattr(c,"coef_"): vals=np.abs(c.coef_[0])
    elif hasattr(c,"feature_importances_"): vals=c.feature_importances_
    else: return {}
    return {n:float(v) for n,v in sorted(zip(names,vals),key=lambda z:z[1],reverse=True)}
def train_and_evaluate(df:pd.DataFrame,model_name:str,features:list[str],test_size:float=TEST_SIZE,random_state:int=RANDOM_STATE):
    validate_dataset(df); validate_features(features)
    if not 0<test_size<1: raise ValueError("test_size must be between 0 and 1")
    X=df[features]; y=df[TARGET]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,stratify=y,random_state=random_state)
    pipe=build_pipeline(model_name,features); pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte); score=pipe.predict_proba(Xte)[:,1]
    metrics = evaluate(yte.to_numpy(), pred, score)
    return pipe, AnalysisResult(
        model_name=model_name,
        features=features,
        metrics=metrics,
        feature_importance=importance(pipe),
        )
