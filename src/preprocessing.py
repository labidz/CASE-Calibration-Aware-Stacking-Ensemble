"""
preprocessing.py
Preprocessing pipelines for all baseline models.

Design principles
-----------------
- All transformations are fitted on the training fold and applied to the
  validation fold. No information from the validation fold influences any
  fitted transformer. This prevents data leakage.
- Tree-based models (RF, XGBoost, LightGBM, CatBoost) receive unscaled
  features; they are invariant to monotonic transformations.
- Distance-based and linear models (LR, SVM, kNN, MLP) receive
  StandardScaler-normalised features.
- For k-Nearest Neighbours, SMOTE is applied inside the training fold.
- class_weight='balanced' is set for all models that support it.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Models that require feature scaling
SCALE_MODELS = {"LR", "SVM", "kNN", "MLP"}

# Models that support class_weight parameter natively
WEIGHT_MODELS = {"LR", "SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP"}


def build_pipeline(
    clf,
    model_name: str,
    multiclass: bool,
    seed: int = 42,
) -> Pipeline:
    """Build a preprocessing + classifier pipeline for the given model.

    Parameters
    ----------
    clf : sklearn-compatible estimator
    model_name : str — key used to determine scaling / SMOTE behaviour
    multiclass : bool — True if the task has more than 2 classes
    seed : int — random seed for SMOTE

    Returns
    -------
    sklearn Pipeline or imblearn Pipeline
    """
    imputer = SimpleImputer(strategy="median")
    needs_scale = model_name in SCALE_MODELS

    # Apply balanced class weights where supported (binary tasks)
    if model_name in WEIGHT_MODELS and not multiclass:
        try:
            clf.set_params(class_weight="balanced")
        except ValueError:
            pass  # model does not accept class_weight (e.g. XGBoost uses scale_pos_weight)

    if model_name == "kNN":
        # kNN does not support class_weight → use SMOTE inside training fold
        k_neighbors = 3  # conservative; avoids failure on very small minority classes
        steps = [
            ("smote", SMOTE(random_state=seed, k_neighbors=k_neighbors)),
            ("imp",   imputer),
            ("sc",    StandardScaler()),
            ("clf",   clf),
        ]
        return ImbPipeline(steps)

    steps = [("imp", imputer)]
    if needs_scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)
