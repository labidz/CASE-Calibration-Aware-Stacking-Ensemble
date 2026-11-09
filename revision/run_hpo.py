"""
run_hpo.py -- nested cross-validation with random-search hyper-parameter
optimisation for every model (Protocol A).

Outer loop : the first 5-fold split of the main benchmark (seed 42), so the
             tuned and default results are paired fold-by-fold.
Inner loop : random search over N_ITER configurations (configuration 0 is the
             default of Table A1), each scored by 3-fold stratified CV
             log-loss on the outer training fold only.
CASE-tuned : CASE / CASE-H6 built from the tuned base models of that outer
             fold; OOF meta-features use a different inner split (seed+100).
"""
import argparse
import os
import pickle
import time
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import log_loss

import data
from pipeline import (BASE_MODELS, DEFAULTS, TREES, HET6, SEED, fit_predict,
                      oof_predict, stack_predict)

OUT = os.environ.get("CASE_OUT", "results")
N_ITER = 10

SPACES = {
    "LR":  dict(C=loguniform(1e-3, 1e2), max_iter=[1000]),
    "SVM": dict(C=loguniform(1e-2, 1e3), gamma=loguniform(1e-4, 1e0), kernel=["rbf"]),
    "kNN": dict(n_neighbors=randint(1, 31), weights=["uniform", "distance"], p=[1, 2]),
    "RF":  dict(n_estimators=[200], max_depth=[None, 5, 10, 20],
                max_features=["sqrt", "log2", 0.5, 1.0], min_samples_leaf=[1, 2, 4, 8]),
    "XGBoost": dict(n_estimators=[100, 200, 400], max_depth=randint(2, 9),
                    learning_rate=loguniform(1e-2, 0.3), subsample=uniform(0.5, 0.5),
                    colsample_bytree=uniform(0.5, 0.5), min_child_weight=loguniform(1, 10),
                    reg_lambda=loguniform(1e-3, 10)),
    "LightGBM": dict(n_estimators=[100, 200, 400], num_leaves=randint(4, 64),
                     learning_rate=loguniform(1e-2, 0.3), subsample=uniform(0.5, 0.5),
                     subsample_freq=[1], colsample_bytree=uniform(0.5, 0.5),
                     min_child_samples=randint(5, 50), reg_lambda=loguniform(1e-3, 10)),
    "CatBoost": dict(iterations=[200, 500], depth=randint(3, 9),
                     learning_rate=loguniform(1e-2, 0.3), l2_leaf_reg=loguniform(1, 10)),
    "MLP": dict(hidden_layer_sizes=[(64,), (128,), (128, 64), (256, 128)],
                alpha=loguniform(1e-5, 1e-1), learning_rate_init=loguniform(1e-4, 1e-2),
                max_iter=[500]),
}


def candidates(model):
    rng = np.random.RandomState(abs(hash(model)) % (2 ** 31) if False else
                                sum(map(ord, model)))
    c = [dict(DEFAULTS[model])]
    c += list(ParameterSampler(SPACES[model], n_iter=N_ITER - 1, random_state=rng))
    for d in c:  # json-friendly
        for k, v in list(d.items()):
            if isinstance(v, np.generic):
                d[k] = v.item()
    return c


def tune(model, Xtr, ytr, cats, X, K):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(cv.split(Xtr, ytr))
    scores = []
    for params in candidates(model):
        ll = []
        for a, b in splits:
            P, _, _ = fit_predict(model, Xtr.iloc[a], ytr[a], Xtr.iloc[b], cats, X,
                                  params=params, K=K)
            ll.append(log_loss(ytr[b], np.clip(P, 1e-15, 1), labels=np.arange(K)))
        scores.append(np.mean(ll))
    i = int(np.argmin(scores))
    return candidates(model)[i], i, scores


def run(name):
    X, y, cats, info = data.load(name)
    K = int(y.max()) + 1
    res = dict(name=name, y=y, folds=[], P={}, best={}, best_idx={}, T={})
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        t0 = time.perf_counter()
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
        res["folds"].append((0, f, tr, te))
        full, best = {}, {}
        for m in BASE_MODELS:
            ts = time.perf_counter()
            params, idx, _ = tune(m, Xtr, ytr, cats, X, K)
            P, tf, _ = fit_predict(m, Xtr, ytr, Xte, cats, X, params=params, K=K)
            full[m], best[m] = P, params
            res["P"].setdefault(m, []).append(P)
            res["best"].setdefault(m, []).append(params)
            res["best_idx"].setdefault(m, []).append(idx)
            res["T"].setdefault(m, []).append(time.perf_counter() - ts)
        oof = {}
        for m in HET6:
            oof[m], _ = oof_predict(m, Xtr, ytr, cats, X, params=best[m], seed=SEED + 100)
        for members, lab in ((TREES, "CASE"), (HET6, "CASE-H6")):
            Mtr = np.hstack([oof[m] for m in members])
            Mte = np.hstack([full[m] for m in members])
            P, _ = stack_predict(Mtr, ytr, Mte, calib="sigmoid")
            res["P"].setdefault(lab, []).append(P)
        print(f"  HPO {name:<12} fold{f} [{time.perf_counter() - t0:6.1f}s]", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for d in a.datasets:
        fn = os.path.join(OUT, f"hpo_A_{d}.pkl")
        lock = fn + ".lock"
        if os.path.exists(fn) or os.path.exists(lock):
            continue
        open(lock, "w").close()
        t0 = time.perf_counter()
        r = run(d)
        r["wall"] = time.perf_counter() - t0
        pickle.dump(r, open(fn, "wb"))
        print(f"DONE HPO {d} in {r['wall']:.0f}s", flush=True)
