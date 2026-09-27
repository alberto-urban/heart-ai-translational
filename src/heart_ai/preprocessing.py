from collections.abc import Sequence
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .config import NUMERICAL,CATEGORICAL
def build_preprocessor(features:Sequence[str])->ColumnTransformer:
    s=set(features); num=[x for x in NUMERICAL if x in s]; cat=[x for x in CATEGORICAL if x in s]; steps=[]
    if num: steps.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num))
    if cat: steps.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    if not steps: raise ValueError("No valid features selected")
    return ColumnTransformer(steps,remainder="drop",verbose_feature_names_out=False)
