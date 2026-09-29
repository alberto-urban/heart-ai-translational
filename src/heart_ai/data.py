import logging
from pathlib import Path

import pandas as pd

from .config import ALL_FEATURES, TARGET

logger=logging.getLogger(__name__)
class DataValidationError(ValueError): pass
def validate_dataset(df:pd.DataFrame)->None:
    missing=set(ALL_FEATURES+[TARGET])-set(df.columns)
    if missing: raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    if df.empty: raise DataValidationError("Dataset is empty")
    if df[TARGET].isna().any(): raise DataValidationError("Target contains missing values")
    if not set(df[TARGET].unique()).issubset({0,1}): raise DataValidationError("Target must contain only 0/1")
def load_dataset(path:Path)->pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Dataset not found at {path}. Download heart.csv from Kaggle into data/")
    logger.info("Loading %s",path); df=pd.read_csv(path); validate_dataset(df); logger.info("Loaded %d rows",len(df)); return df
def validate_features(features:list[str])->None:
    if not features: raise ValueError("At least one feature must be selected")
    unknown=set(features)-set(ALL_FEATURES)
    if unknown: raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
