"""
case.py
CASE: Calibration-Aware Stacking Ensemble

Architecture (3 layers)
-----------------------
Layer 1 — Base Models (pre-specified, tree-based)
    CatBoost, Random Forest, LightGBM, XGBoost.
    Selected BEFORE any experiments based on prior literature consensus
    (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022) that tree-based
    models dominate heterogeneous tabular benchmarks. This pre-specification
    avoids any circular dependency between base model selection and benchmark
    results.

Layer 2 — Meta-Learner
    Logistic Regression trained on out-of-fold (OOF) probability
    concatenations from all four base models, obtained via 3-fold inner
    cross-validation on the outer training fold.

Layer 3 — Calibration
    Platt scaling (CalibratedClassifierCV, method='sigmoid') fitted via
    3-fold cross-validation on the outer training fold. No held-out
    calibration set is used; all calibration is done within the training
    fold to avoid leakage.

Usage
-----
    from src.case import run_case_fold
    probs, metrics = run_case_fold(X_train, y_train, X_val, y_val, multiclass)
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .evaluation import compute_metrics

SEED = 42
INNER_FOLDS = 3  # inner CV folds for OOF and calibration


def _make_base_models(multiclass: bool) -> list:
    """Return fresh instances of the four pre-specified base models."""
    # Note: class_weight is not applied for multiclass (not universally supported)
    cw = "balanced" if not multiclass else None
    return [
        ("CatBoost", CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.10,
            random_state=SEED, verbose=0)),
        ("RF",       RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight=cw,
            random_state=SEED, n_jobs=-1)),
        ("LightGBM", LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.10,
            class_weight=cw, random_state=SEED, n_jobs=-1, verbose=-1)),
        ("XGBoost",  XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.10,
            eval_metric="mlogloss", random_state=SEED, n_jobs=-1, verbosity=0)),
    ]


def _build_base_pipe(clf, needs_impute: bool = True) -> Pipeline:
    steps = [("imp", SimpleImputer(strategy="median")), ("clf", clf)]
    return Pipeline(steps)


def run_case_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    multiclass: bool,
    seed: int = SEED,
) -> tuple:
    """Run CASE on a single outer CV fold.

    Parameters
    ----------
    X_train, y_train : outer training data
    X_val,   y_val   : outer validation data
    multiclass       : True if n_classes > 2
    seed             : random seed

    Returns
    -------
    y_pred  : np.ndarray of predicted class labels
    y_proba : np.ndarray of calibrated class probabilities
    metrics : dict with acc, f1, auc, mcc, pre, rec
    """
    n_classes = len(np.unique(y_train))
    base_models = _make_base_models(multiclass)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)

    # ── Layer 1 + 2: collect OOF and validation probabilities ────────────────
    meta_train_blocks = []
    meta_val_blocks   = []

    for bm_name, bm_clf in base_models:
        oof_probs = np.zeros((len(X_train), n_classes))
        val_probs = np.zeros((len(X_val),   n_classes))

        # OOF predictions via inner CV
        for tr_idx, vl_idx in inner_cv.split(X_train, y_train):
            pipe = _build_base_pipe(bm_clf.__class__(**bm_clf.get_params()))
            try:
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                oof_probs[vl_idx] = pipe.predict_proba(X_train[vl_idx])
            except Exception:
                pass  # leave oof_probs[vl_idx] as zeros

        # Retrain on full outer training fold for val predictions
        pipe_full = _build_base_pipe(bm_clf.__class__(**bm_clf.get_params()))
        try:
            pipe_full.fit(X_train, y_train)
            val_probs = pipe_full.predict_proba(X_val)
        except Exception:
            pass

        meta_train_blocks.append(oof_probs)
        meta_val_blocks.append(val_probs)

    M_train = np.hstack(meta_train_blocks)   # shape: (n_train, 4*K)
    M_val   = np.hstack(meta_val_blocks)     # shape: (n_val,   4*K)

    # ── Layer 2: meta-learner (Logistic Regression) ───────────────────────────
    meta_lr = LogisticRegression(max_iter=1000, random_state=seed)

    # ── Layer 3: Platt calibration via 3-fold CV on training fold ─────────────
    calibrated_meta = CalibratedClassifierCV(
        estimator=meta_lr,
        cv=INNER_FOLDS,
        method="sigmoid",
    )

    try:
        calibrated_meta.fit(M_train, y_train)
        y_pred  = calibrated_meta.predict(M_val)
        y_proba = calibrated_meta.predict_proba(M_val)
    except Exception:
        # Fallback: uncalibrated meta-learner
        meta_lr.fit(M_train, y_train)
        y_pred  = meta_lr.predict(M_val)
        y_proba = meta_lr.predict_proba(M_val)

    metrics = compute_metrics(y_val, y_pred, y_proba, multiclass)
    return y_pred, y_proba, metrics
