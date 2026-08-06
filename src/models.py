"""
models.py
Factory functions for all eight baseline models with fixed best-practice
hyperparameters.

Hyperparameter philosophy
--------------------------
All hyperparameters are fixed to well-established defaults from the respective
model literature. No per-dataset tuning was performed; this ensures that
relative performance reflects inherent model capabilities rather than the
outcome of model-specific optimisation effort.

To reproduce paper results exactly, do NOT change these values.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


SEED = 42  # Global random seed for all models


def make_lr() -> LogisticRegression:
    """Logistic Regression — L2 regularisation, C=1.0."""
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
        multi_class="auto",
    )


def make_svm() -> SVC:
    """Support Vector Machine — RBF kernel, C=10, gamma=scale.
    Probability estimates via Platt calibration (probability=True)."""
    return SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=SEED,
    )


def make_knn() -> KNeighborsClassifier:
    """k-Nearest Neighbours — k=5, Euclidean distance.
    Class imbalance handled via SMOTE (see preprocessing.py)."""
    return KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
    )


def make_rf() -> RandomForestClassifier:
    """Random Forest — 200 trees, max depth 10."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
    )


def make_xgboost() -> XGBClassifier:
    """XGBoost — 200 rounds, depth 5, learning rate 0.10."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.10,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def make_lightgbm() -> LGBMClassifier:
    """LightGBM — 200 rounds, 31 leaves, learning rate 0.10."""
    return LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.10,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def make_catboost() -> CatBoostClassifier:
    """CatBoost — 200 iterations, depth 5, learning rate 0.10."""
    return CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.10,
        random_state=SEED,
        verbose=0,
    )


def make_mlp() -> MLPClassifier:
    """Multi-Layer Perceptron — hidden layers (128, 64), L2 alpha=0.01."""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.01,
        max_iter=500,
        random_state=SEED,
    )


# Registry: ordered dict of model name → factory function
MODEL_REGISTRY = {
    "LR":       make_lr,
    "SVM":      make_svm,
    "kNN":      make_knn,
    "RF":       make_rf,
    "XGBoost":  make_xgboost,
    "LightGBM": make_lightgbm,
    "CatBoost": make_catboost,
    "MLP":      make_mlp,
}

# Display order for tables and figures
MODEL_ORDER = ["LR", "SVM", "kNN", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP", "CASE"]

# Color palette for figures
MODEL_COLORS = {
    "LR":       "#6B9BD1",
    "SVM":      "#F0A500",
    "kNN":      "#E07070",
    "RF":       "#5BAD72",
    "XGBoost":  "#A569BD",
    "LightGBM": "#45B8AC",
    "CatBoost": "#E86E3A",
    "MLP":      "#85929E",
    "CASE":     "#2C3E50",
}
