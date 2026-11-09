"""stats.py -- multiple-dataset comparison statistics (Demsar 2006; Garcia &
Herrera 2008; Benavoli et al. 2016)."""
import numpy as np
from scipy import stats

# Studentised range statistic divided by sqrt(2), alpha = 0.05 (Demsar 2006, Tab. 5a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}


def ranks(S, higher_better=True):
    """S: (N datasets, k models) -> rank matrix (1 = best, ties averaged)."""
    S = np.asarray(S, float)
    return np.vstack([stats.rankdata(-r if higher_better else r, method="average") for r in S])


def friedman(S, higher_better=True):
    S = np.asarray(S, float)
    N, k = S.shape
    R = ranks(S, higher_better)
    chi2, p = stats.friedmanchisquare(*S.T)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    return dict(chi2=chi2, p=p, df=k - 1, ff=ff, p_ff=p_ff, df1=k - 1,
                df2=(k - 1) * (N - 1), avg_rank=R.mean(0), N=N, k=k)


def nemenyi_cd(k, N):
    return Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation of a - b (Kerby 2014).
    +1: a better on every dataset; -1: b better on every dataset."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return (rp - rm) / (rp + rm)


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def wilcoxon_vs(ref, others, names, higher_better=True):
    """Two-sided Wilcoxon signed-rank tests of ref against each column of others."""
    rows = []
    for j, nm in enumerate(names):
        a, b = np.asarray(ref, float), np.asarray(others[:, j], float)
        d = a - b if higher_better else b - a  # positive = ref better
        nz = np.abs(d) > 1e-12
        try:
            w = stats.wilcoxon(a[nz], b[nz], zero_method="wilcox")
            W, p = w.statistic, w.pvalue
        except ValueError:
            W, p = np.nan, 1.0
        rows.append(dict(model=nm, mean_diff=d.mean(), median_diff=np.median(d),
                         ci=boot_ci(d), wins=int((d > 1e-12).sum()),
                         ties=int((~nz).sum()), losses=int((d < -1e-12).sum()),
                         W=W, p=p, r_rb=rank_biserial(a, b) if higher_better else rank_biserial(b, a)))
    adjh = holm([r["p"] for r in rows])
    for r, h in zip(rows, adjh):
        r["p_holm"] = h
        r["p_bonf"] = min(1.0, r["p"] * len(rows))
    return rows
