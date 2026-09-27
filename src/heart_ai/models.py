from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .config import RANDOM_STATE
MODEL_NAMES=["logistic_regression","random_forest","svm"]
def build_model(name:str):
    if name=="logistic_regression": return LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RANDOM_STATE)
    if name=="random_forest": return RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1)
    if name == "svm": 
        svm = SVC(class_weight="balanced", random_state=42) 
        return CalibratedClassifierCV(estimator=svm, method="sigmoid", cv=5)
    raise ValueError(f"Unknown model '{name}'. Supported: {MODEL_NAMES}")
