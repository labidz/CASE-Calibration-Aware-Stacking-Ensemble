"""
run_bench.py -- repeated stratified 5-fold CV for all baselines, CASE and the
ablation variants.  All variants within a fold share the same base-model fits
(full-fold fits for validation predictions, 3-fold inner fits for OOF
predictions), so ablations differ only in the component under study.

usage: python run_bench.py --datasets Heart WDBC ... --protocol A --repeats 2
"""
import argparse
import os
import pickle
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import data
from pipeline import (BASE_MODELS, OOF_MODELS, TREES, HET4, HET6, SEED,
                      fit_predict, oof_predict, stack_predict, post_calibrate)

OUT = os.environ.get("CASE_OUT", "results")


def run_dataset(name, protocol="A", repeats=2, save_oof=False):
    balanced = protocol == "B"
    X, y, cats, info = data.load(name)
    K = int(y.max()) + 1
    res = dict(name=name, info=info, y=y, protocol=protocol, folds=[], P={}, T={},
               oof_corr=[], oof_err_corr=[], oof=[] if save_oof else None, select=[])
    def add(model, P, t):
        res["P"].setdefault(model, []).append(P)
        res["T"].setdefault(model, []).append(t)

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for f, (tr, te) in enumerate(skf.split(X, y)):
            t_start = time.perf_counter()
            Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
            res["folds"].append((rep, f, tr, te))
            full, tfull = {}, {}
            for m in ["Majority"] + BASE_MODELS:
                P, tf, tp = fit_predict(m, Xtr, ytr, Xte, cats, X, balanced=balanced, K=K)
                full[m], tfull[m] = P, tf
                add(m, P, tf)
            oof, toof = {}, {}
            for m in OOF_MODELS:
                oof[m], toof[m] = oof_predict(m, Xtr, ytr, cats, X, balanced=balanced)
            # diversity diagnostics: correlation of OOF probability of the true class
            pt = np.column_stack([oof[m][np.arange(len(ytr)), ytr] for m in OOF_MODELS])
            res["oof_corr"].append(np.corrcoef(pt.T))
            err = np.column_stack([(oof[m].argmax(1) != ytr).astype(float) for m in OOF_MODELS])
            with np.errstate(invalid="ignore", divide="ignore"):
                res["oof_err_corr"].append(np.corrcoef(err.T))
            if save_oof:
                res["oof"].append({m: oof[m] for m in OOF_MODELS})

            def stack(members, calib, label, meta_bal=balanced):
                Mtr = np.hstack([oof[m] for m in members])
                Mte = np.hstack([full[m] for m in members])
                P, tm = stack_predict(Mtr, ytr, Mte, calib=calib, balanced=meta_bal)
                cost = sum(toof[m] + tfull[m] for m in members) + tm
                add(label, P, cost)

            stack(TREES, "sigmoid", "CASE")          # original CASE
            stack(TREES, None, "Stack")              # ablation: no calibration
            stack(TREES, "isotonic", "CASE-Iso")     # ablation: isotonic calibration
            stack(HET4, "sigmoid", "CASE-H4")        # heterogeneous base set
            stack(HET6, "sigmoid", "CASE-H6")        # all six base models
            # soft voting over the four tree models (no meta-learner, no calibration)
            add("SoftVote", np.mean([full[m] for m in TREES], 0),
                sum(tfull[m] for m in TREES))
            # single calibrated CatBoost (calibrator fitted on its OOF predictions)
            for meth, lab in (("sigmoid", "CatBoost+Platt"), ("isotonic", "CatBoost+Iso")):
                add(lab, post_calibrate(oof["CatBoost"], ytr, full["CatBoost"], meth),
                    toof["CatBoost"] + tfull["CatBoost"])
            # select-best: single base model with the lowest OOF log-loss
            ll = {m: log_loss(ytr, np.clip(oof[m], 1e-15, 1), labels=np.arange(K))
                  for m in OOF_MODELS}
            best = min(ll, key=ll.get)
            res["select"].append(best)
            add("SelectBest", full[best], sum(toof.values()) + tfull[best])
            print(f"  {name:<22} {protocol} rep{rep} fold{f} "
                  f"[{time.perf_counter() - t_start:6.1f}s] best={best}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tag", default="core")
    ap.add_argument("--save-oof", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for d in a.datasets:
        fn = os.path.join(OUT, f"{a.tag}_{a.protocol}_{d}.pkl")
        if os.path.exists(fn):
            print("skip", fn); continue
        t0 = time.perf_counter()
        r = run_dataset(d, a.protocol, a.repeats, a.save_oof)
        r["wall"] = time.perf_counter() - t0
        with open(fn, "wb") as fh:
            pickle.dump(r, fh)
        print(f"DONE {d} {a.protocol} in {r['wall']:.0f}s", flush=True)
