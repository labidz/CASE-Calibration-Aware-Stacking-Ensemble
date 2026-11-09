"""
run_automl.py -- AutoGluon-Tabular and TabPFN (v1) on the core suite, using
exactly the same outer folds as run_bench.py (first N_REP repeats).
Run with the separate virtual environment that contains torch/AutoGluon.
"""
import argparse
import os
import pickle
import shutil
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import data
from pipeline import make_preprocessor, SEED

warnings.filterwarnings("ignore")
OUT = os.environ.get("CASE_OUT", "results")


def run_tabpfn(Xtr, ytr, Xte, cats, X, K):
    from tabpfn import TabPFNClassifier
    pre = make_preprocessor("RF", list(X.columns), cats, X)   # impute + ordinal codes
    A, B = pre.fit_transform(Xtr), pre.transform(Xte)
    clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=16, seed=SEED)
    clf.fit(A, ytr)
    P = clf.predict_proba(B)
    out = np.zeros((len(Xte), K)); out[:, clf.classes_.astype(int)] = P
    return out


def run_autogluon(Xtr, ytr, Xte, cats, K, time_limit):
    from autogluon.tabular import TabularPredictor
    tr = Xtr.copy(); te = Xte.copy()
    for c in cats:
        tr[c] = tr[c].astype("category"); te[c] = te[c].astype("category")
    tr["__y"] = ytr
    path = tempfile.mkdtemp(prefix="ag_")
    pred = TabularPredictor(label="__y", path=path, verbosity=0,
                            problem_type="binary" if K == 2 else "multiclass")
    pred.fit(tr, presets="medium_quality", time_limit=time_limit)
    P = pred.predict_proba(te).values
    cls = np.array(pred.class_labels, dtype=int)
    out = np.zeros((len(Xte), K)); out[:, cls] = P
    shutil.rmtree(path, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--method", choices=["TabPFN", "AutoGluon"], required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--time-limit", type=int, default=30)
    a = ap.parse_args()
    for d in a.datasets:
        fn = os.path.join(OUT, f"automl_{a.method}_{d}.pkl")
        if os.path.exists(fn):
            continue
        X, y, cats, info = data.load(d)
        K = int(y.max()) + 1
        res = dict(name=d, y=y, folds=[], P={a.method: []}, T={a.method: []})
        for rep in range(a.repeats):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
            for f, (tr, te) in enumerate(skf.split(X, y)):
                t0 = time.perf_counter()
                if a.method == "TabPFN":
                    P = run_tabpfn(X.iloc[tr], y[tr], X.iloc[te], cats, X, K)
                else:
                    P = run_autogluon(X.iloc[tr], y[tr], X.iloc[te], cats, K, a.time_limit)
                res["folds"].append((rep, f, tr, te))
                res["P"][a.method].append(P)
                res["T"][a.method].append(time.perf_counter() - t0)
                print(f"  {a.method} {d} rep{rep} fold{f} acc={(P.argmax(1)==y[te]).mean():.3f} "
                      f"[{res['T'][a.method][-1]:.1f}s]", flush=True)
        pickle.dump(res, open(fn, "wb"))
        print("DONE", a.method, d, flush=True)
