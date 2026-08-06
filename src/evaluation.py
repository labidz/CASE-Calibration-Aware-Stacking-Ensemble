"""
evaluation.py
Evaluation protocol: repeated stratified k-fold cross-validation.
Computes Accuracy, Macro-F1, AUC-ROC, and MCC for each model-dataset pair.

Protocol
--------
- 5-fold stratified cross-validation, repeated 2× with different seeds
- Total: 10 evaluation runs per model per dataset
- All preprocessing inside each fold (no leakage)
- Metrics averaged across all 10 runs; std reported alongside mean
"""

import time
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from .preprocessing import build_pipeline

warnings.filterwarnings("ignore")

N_FOLDS   = 5
N_REPEATS = 2
SEED      = 42

METRIC_KEYS = ["acc", "f1", "auc", "mcc", "pre", "rec"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    multiclass: bool,
) -> dict:
    """Compute all six evaluation metrics for a single fold.

    Returns
    -------
    dict with keys: acc, f1, auc, mcc, pre, rec
    """
    avg = "macro" if multiclass else "binary"

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)

    try:
        if multiclass:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        else:
            auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        auc = np.nan

    return {"acc": acc, "f1": f1, "auc": auc, "mcc": mcc, "pre": pre, "rec": rec}


def evaluate_model(
    model_name: str,
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    multiclass: bool,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Run repeated stratified k-fold CV for a single model on a single dataset.

    Parameters
    ----------
    model_name    : display name (key into MODEL_REGISTRY)
    model_factory : callable that returns a fresh model instance
    X, y          : feature matrix and label array
    multiclass    : True if n_classes > 2
    seed          : base random seed
    verbose       : print per-model summary

    Returns
    -------
    dict mapping metric key → list of fold values (length N_FOLDS × N_REPEATS)
    """
    fold_results = {k: [] for k in METRIC_KEYS}
    t0 = time.perf_counter()

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + rep)
        for tr_idx, vl_idx in skf.split(X, y):
            clf  = model_factory()
            pipe = build_pipeline(clf, model_name, multiclass, seed)
            try:
                pipe.fit(X[tr_idx], y[tr_idx])
                y_pred  = pipe.predict(X[vl_idx])
                y_proba = pipe.predict_proba(X[vl_idx])
                m = compute_metrics(y[vl_idx], y_pred, y_proba, multiclass)
            except Exception:
                m = {k: np.nan for k in METRIC_KEYS}
            for k in METRIC_KEYS:
                fold_results[k].append(m[k])

    elapsed = time.perf_counter() - t0
    if verbose:
        a = np.nanmean(fold_results["acc"])
        f = np.nanmean(fold_results["f1"])
        print(f"    {model_name:<10}  acc={a:.4f}  f1={f:.4f}  [{elapsed:.1f}s]")

    return fold_results


def evaluate_case(
    X: np.ndarray,
    y: np.ndarray,
    multiclass: bool,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Run repeated stratified k-fold CV for CASE on a single dataset.

    Imports run_case_fold lazily to avoid circular imports.
    """
    from .case import run_case_fold

    fold_results = {k: [] for k in METRIC_KEYS}
    t0 = time.perf_counter()

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + rep)
        for tr_idx, vl_idx in skf.split(X, y):
            try:
                _, _, m = run_case_fold(
                    X[tr_idx], y[tr_idx],
                    X[vl_idx], y[vl_idx],
                    multiclass, seed=seed,
                )
            except Exception:
                m = {k: np.nan for k in METRIC_KEYS}
            for k in METRIC_KEYS:
                fold_results[k].append(m[k])

    elapsed = time.perf_counter() - t0
    if verbose:
        a = np.nanmean(fold_results["acc"])
        f = np.nanmean(fold_results["f1"])
        print(f"    {'CASE':<10}  acc={a:.4f}  f1={f:.4f}  [{elapsed:.1f}s]")

    return fold_results


def summarise(fold_results: dict) -> dict:
    """Compute mean and std for each metric across all folds."""
    return {
        k: {
            "mean": float(np.nanmean(fold_results[k])),
            "std":  float(np.nanstd(fold_results[k])),
        }
        for k in METRIC_KEYS
    }
