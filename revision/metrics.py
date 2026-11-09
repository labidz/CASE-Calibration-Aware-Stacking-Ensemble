"""
metrics.py -- per-fold metrics computed from stored class-probability matrices.

Predicted label = argmax of the class-probability vector (identical rule for
every model).

ECE (top-label, Guo et al. 2017; Naeini et al. 2015), M = 10 equal-width bins:
    ECE = sum_m |B_m|/n * | acc(B_m) - conf(B_m) |
with conf = max_k p_k and acc(B_m) = fraction of correct predictions in bin m.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, matthews_corrcoef, log_loss, recall_score)

N_BINS = 10


def ece_toplabel(y, P, n_bins=N_BINS):
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def reliability(y, P, n_bins=N_BINS):
    """Top-label reliability curve: (mean confidence, accuracy, count) per bin."""
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            out.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(out)


def fold_metrics(y, P):
    K = P.shape[1]
    labels = np.arange(K)
    Pn = np.clip(P, 1e-15, 1)
    Pn = Pn / Pn.sum(1, keepdims=True)
    yhat = P.argmax(1)
    if K == 2:
        try:
            auc = roc_auc_score(y, P[:, 1])
        except ValueError:
            auc = np.nan
    else:
        try:
            auc = roc_auc_score(y, Pn, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            auc = np.nan
    Y = np.eye(K)[y]
    return dict(
        acc=accuracy_score(y, yhat),
        bacc=balanced_accuracy_score(y, yhat),
        f1=f1_score(y, yhat, average="macro", labels=labels, zero_division=0),
        auc=auc,
        mcc=matthews_corrcoef(y, yhat),
        ll=log_loss(y, Pn, labels=labels),
        brier=np.mean(np.sum((P - Y) ** 2, 1)),
        ece=ece_toplabel(y, P),
        recall=recall_score(y, yhat, average=None, labels=labels, zero_division=0),
    )


_MEMO = {}


def all_fold_metrics(res, model):
    """List of metric dicts, one per outer fold (memoised)."""
    key = (id(res), res["name"], res.get("protocol"), model, len(res["P"][model]))
    if key not in _MEMO:
        y = res["y"]
        _MEMO[key] = [fold_metrics(y[te], P) for P, (_, _, _, te) in zip(res["P"][model], res["folds"])]
    return _MEMO[key]


def summary(res, model, keys=("acc", "bacc", "f1", "auc", "mcc", "ll", "brier", "ece")):
    fm = all_fold_metrics(res, model)
    out = {k: (np.nanmean([m[k] for m in fm]), np.nanstd([m[k] for m in fm])) for k in keys}
    out["recall"] = np.mean([m["recall"] for m in fm], 0)
    return out
