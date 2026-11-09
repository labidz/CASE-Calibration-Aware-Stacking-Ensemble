"""
pipeline.py -- models, preprocessing, calibration and stacking utilities.

Protocol A (primary): no class-imbalance correction for any model.
Protocol B (sensitivity): class-balanced weighting for every model
  (class_weight / sample_weight; SMOTE with adaptive k for kNN, which has no
  weighting mechanism).
"""
import time
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 42
INNER_FOLDS = 3
SCALE_MODELS = {"LR", "SVM", "kNN", "MLP"}
BASE_MODELS = ["LR", "SVM", "kNN", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP"]
OOF_MODELS = ["LR", "SVM", "RF", "XGBoost", "LightGBM", "CatBoost"]
TREES = ["CatBoost", "RF", "LightGBM", "XGBoost"]          # CASE base set
HET4 = ["CatBoost", "RF", "SVM", "LR"]                      # heterogeneous set
HET6 = ["CatBoost", "RF", "LightGBM", "XGBoost", "SVM", "LR"]

# default ("fixed") hyper-parameters, identical to the original submission
DEFAULTS = {
    "LR":       dict(C=1.0, max_iter=1000),
    "SVM":      dict(C=10.0, kernel="rbf", gamma="scale"),
    "kNN":      dict(n_neighbors=5, metric="euclidean"),
    "RF":       dict(n_estimators=200, max_depth=10),
    "XGBoost":  dict(n_estimators=200, max_depth=5, learning_rate=0.10),
    "LightGBM": dict(n_estimators=200, num_leaves=31, learning_rate=0.10),
    "CatBoost": dict(iterations=200, depth=5, learning_rate=0.10),
    "MLP":      dict(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=500),
}


class KNNSmote(BaseEstimator, ClassifierMixin):
    """kNN with optional SMOTE (adaptive k so that very small classes work)."""

    def __init__(self, n_neighbors=5, weights="uniform", p=2, metric="minkowski",
                 smote=False, random_state=SEED):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.smote = smote
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.smote:
            from imblearn.over_sampling import SMOTE
            n_min = np.bincount(y).min()
            if n_min >= 2:
                k = int(min(5, n_min - 1))
                X, y = SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X, y)
        metric = self.metric if self.metric != "minkowski" else "minkowski"
        self.knn_ = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(y)),
                                         weights=self.weights, p=self.p, metric=metric)
        self.knn_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.knn_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


def make_estimator(name, params=None, balanced=False, n_classes=2, seed=SEED):
    p = dict(DEFAULTS.get(name, {})) if params is None else dict(params)
    if name == "LR":
        return LogisticRegression(random_state=seed,
                                  class_weight="balanced" if balanced else None, **p)
    if name == "SVM":
        return SVC(probability=True, random_state=seed,
                   class_weight="balanced" if balanced else None, **p)
    if name == "kNN":
        q = dict(p)
        if q.get("metric") == "euclidean":
            q["metric"] = "minkowski"; q.setdefault("p", 2)
        return KNNSmote(smote=balanced, random_state=seed, **q)
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=1,
                                      class_weight="balanced" if balanced else None, **p)
    if name == "XGBoost":
        return XGBClassifier(random_state=seed, n_jobs=1, verbosity=0,
                             tree_method="hist", **p)        # weights via sample_weight
    if name == "LightGBM":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1,
                              class_weight="balanced" if balanced else None, **p)
    if name == "CatBoost":
        return CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0,
                                  allow_writing_files=False,
                                  auto_class_weights="Balanced" if balanced else None, **p)
    if name == "MLP":
        return MLPClassifier(random_state=seed, **p)      # weights via sample_weight
    if name == "Majority":
        return DummyClassifier(strategy="prior")
    raise KeyError(name)


def make_preprocessor(name, columns, cats, X):
    """Numeric: median imputation (+ standardisation for scale-sensitive models).
    Categorical: mode imputation; one-hot for scale-sensitive models,
    integer (ordinal) codes for tree models."""
    cats = [c for c in cats if c in columns]
    nominal = [c for c in cats if X[c].nunique() > 2]
    binary = [c for c in cats if c not in nominal]
    num = [c for c in columns if c not in cats]
    scale = name in SCALE_MODELS
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)] if num else []
    if binary:
        transformers.append(("bin", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]),
            binary))
    if nominal:
        enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if scale else
               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        transformers.append(("nom", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]), nominal))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class Model:
    """Preprocessing + estimator, with balanced sample weights where needed."""

    def __init__(self, name, cats, X_ref, params=None, balanced=False, seed=SEED, K=None):
        self.K_fixed = K
        self.name, self.cats, self.X_ref = name, cats, X_ref
        self.params, self.balanced, self.seed = params, balanced, seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.K = self.K_fixed or int(y.max()) + 1
        self.pre = make_preprocessor(self.name, list(X.columns), self.cats, self.X_ref)
        Xt = self.pre.fit_transform(X)
        self.est = make_estimator(self.name, self.params, self.balanced, self.K, self.seed)
        kw = {}
        if self.balanced and self.name in ("XGBoost", "MLP"):
            kw["sample_weight"] = compute_sample_weight("balanced", y)
        self.est.fit(Xt, y, **kw)
        return self

    def predict_proba(self, X):
        P = self.est.predict_proba(self.pre.transform(X))
        out = np.zeros((len(X), self.K))
        out[:, np.asarray(self.est.classes_, dtype=int)] = P
        return out


def fit_predict(name, Xtr, ytr, Xte, cats, X_ref, params=None, balanced=False, seed=SEED, K=None):
    t0 = time.perf_counter()
    m = Model(name, cats, X_ref, params, balanced, seed, K).fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    P = m.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return P, t_fit, t_pred


def oof_predict(name, Xtr, ytr, cats, X_ref, params=None, balanced=False, seed=SEED):
    """Out-of-fold class probabilities on the outer training fold (3-fold)."""
    K = int(ytr.max()) + 1
    O = np.zeros((len(ytr), K))
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    t = 0.0
    for a, b in cv.split(Xtr, ytr):
        P, tf, _ = fit_predict(name, Xtr.iloc[a], ytr[a], Xtr.iloc[b], cats, X_ref,
                               params, balanced, seed, K)
        O[b] = P
        t += tf
    return O, t


# ---------------------------------------------------------------- meta level
def _meta_features(blocks, K):
    """Concatenate class-probability blocks; for binary tasks only P(y=1)
    is redundant-free but we keep all K columns as in the original CASE."""
    return np.hstack(blocks)


def stack_predict(M_tr, y_tr, M_te, calib=None, balanced=False, seed=SEED):
    """Logistic-regression meta-learner, optionally wrapped by
    CalibratedClassifierCV(cv=3) (method='sigmoid' = Platt, or 'isotonic')."""
    K = int(y_tr.max()) + 1
    meta = LogisticRegression(max_iter=1000, random_state=seed,
                              class_weight="balanced" if balanced else None)
    t0 = time.perf_counter()
    if calib is None:
        meta.fit(M_tr, y_tr)
        P = meta.predict_proba(M_te)
        cls = meta.classes_
    else:
        cal = CalibratedClassifierCV(estimator=meta, cv=INNER_FOLDS, method=calib)
        cal.fit(M_tr, y_tr)
        P = cal.predict_proba(M_te)
        cls = cal.classes_
    out = np.zeros((len(M_te), K))
    out[:, np.asarray(cls, dtype=int)] = P
    return out, time.perf_counter() - t0


def _platt_fit(f, t):
    """Platt (1999) sigmoid fit with his target smoothing; returns (A, B)."""
    from scipy.optimize import minimize
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    tt = np.where(t == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def nll(ab):
        z = ab[0] * f + ab[1]
        # P = 1/(1+exp(z)) ; loss = -[tt log P + (1-tt) log(1-P)]
        return np.sum(tt * np.logaddexp(0, z) + (1 - tt) * np.logaddexp(0, -z))
    r = minimize(nll, x0=np.array([0.0, np.log((n_neg + 1) / (n_pos + 1))]), method="BFGS")
    return r.x


def post_calibrate(P_oof, y_oof, P_te, method):
    """One-vs-rest Platt or isotonic recalibration of a single model's
    probabilities, fitted on out-of-fold predictions of the training fold
    (equivalent to CalibratedClassifierCV(ensemble=False))."""
    K = P_te.shape[1]
    cols = [1] if K == 2 else list(range(K))
    Q = np.zeros_like(P_te)
    for k in cols:
        t = (y_oof == k).astype(int)
        if method == "sigmoid":
            A, B = _platt_fit(P_oof[:, k], t)
            Q[:, k] = 1.0 / (1.0 + np.exp(A * P_te[:, k] + B))
        else:
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(P_oof[:, k], t)
            Q[:, k] = ir.predict(P_te[:, k])
    if K == 2:
        Q[:, 0] = 1 - Q[:, 1]
    else:
        s = Q.sum(1, keepdims=True)
        Q = np.where(s > 0, Q / np.where(s > 0, s, 1), 1.0 / K)
    return Q
