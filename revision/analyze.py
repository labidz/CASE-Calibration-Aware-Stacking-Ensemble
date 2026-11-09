"""
analyze.py -- all statistics, tables (LaTeX fragments) and figures of the
revised manuscript, computed from the stored per-fold probability matrices.
"""
import glob
import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
import metrics as M
import stats as S

RES = "results"
OUT = "out"
FIG = os.path.join(OUT, "figs")
TAB = os.path.join(OUT, "tables")
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

MAIN = ["LR", "SVM", "kNN", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP", "CASE"]
BASEL = MAIN[:-1]
ABL = ["CASE", "Stack", "CASE-Iso", "SoftVote", "CatBoost", "CatBoost+Platt",
       "CatBoost+Iso", "SelectBest", "CASE-H4", "CASE-H6"]
SHORT = {"Heart": "Heart", "WDBC": "WDBC", "Haberman": "Haberman", "Horse": "Horse Colic",
         "Pima": "Pima", "Ecoli": "E.coli", "Sonar": "Sonar", "German": "German",
         "Glass": "Glass", "Ionosphere": "Ionosphere"}
for k, v in data.EXT_META.items():
    SHORT[k] = v["short"]
METRIC_NAMES = dict(acc="Accuracy", bacc="Balanced accuracy", f1="Macro-F1",
                    auc="AUC-ROC", mcc="MCC", ll="Log-loss", brier="Brier score",
                    ece="ECE")
LOWER = {"ll", "brier", "ece"}
NUM = {}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8c8b86", "#0b0b0b"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def load(pattern):
    out = {}
    for fn in sorted(glob.glob(os.path.join(RES, pattern))):
        r = pickle.load(open(fn, "rb"))
        out[r["name"]] = r
    return out


def table_of(R, order, models, key):
    """mean/std of a metric: DataFrames (datasets x models)."""
    mu = pd.DataFrame(index=order, columns=models, dtype=float)
    sd = mu.copy()
    for d in order:
        for m in models:
            fm = M.all_fold_metrics(R[d], m)
            v = [x[key] for x in fm]
            mu.loc[d, m], sd.loc[d, m] = np.nanmean(v), np.nanstd(v)
    return mu, sd


_cache = {}


def cached_table(tag, R, order, models, key):
    k = (tag, tuple(order), tuple(models), key)
    if k not in _cache:
        _cache[k] = table_of(R, order, models, key)
    return _cache[k]


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if np.isfinite(x) else "--"


def latex_metric_table(mu, sd, key, label, caption, extra_cols=(), rank_row=True,
                       nd=3, names=SHORT):
    models = list(mu.columns)
    lower = key in LOWER
    lines = []
    cols = "l" + "c" * len(models)
    lines.append("\\begin{sidewaystable}")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    lines.append("\\scriptsize\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular*}}{{\\textheight}}{{@{{\\extracolsep\\fill}}{cols}}}")
    lines.append("\\toprule")
    lines.append("Dataset & " + " & ".join(models) + " \\\\")
    lines.append("\\midrule")
    rank_cols = [m for m in models if m in MAIN]
    for d in mu.index:
        row = mu.loc[d, rank_cols].values.astype(float)
        best = np.nanmin(row) if lower else np.nanmax(row)
        cells = []
        for m in models:
            v, s = mu.loc[d, m], sd.loc[d, m]
            c = f"{fmt(v, nd)}$\\pm${fmt(s, 2)[1:] if s < 1 else fmt(s, 2)}"
            if m in rank_cols and abs(v - best) < 5e-5:
                c = f"\\textbf{{{fmt(v, nd)}}}$\\pm${fmt(s, 2)[1:] if s < 1 else fmt(s, 2)}"
            cells.append(c)
        lines.append(f"{names.get(d, d)} & " + " & ".join(cells) + " \\\\")
    if rank_row:
        Rk = S.ranks(mu[rank_cols].values, higher_better=not lower).mean(0)
        best = Rk.min()
        cells = []
        for m in models:
            if m in rank_cols:
                v = Rk[rank_cols.index(m)]
                cells.append(f"\\textbf{{{v:.2f}}}" if abs(v - best) < 1e-9 else f"{v:.2f}")
            else:
                cells.append("--")
        lines.append("\\midrule")
        lines.append("Avg.\\ rank & " + " & ".join(cells) + " \\\\")
    lines.append("\\botrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{sidewaystable}")
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(lines))


# =========================================================================
def dataset_table():
    rows = []
    for d in data.CORE_ORDER + data.EXT_ORDER:
        rows.append(data.describe(d))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "datasets.csv"), index=False)
    cite = {"Heart": "detrano1989", "WDBC": "wolberg1995", "Haberman": "haberman1976",
            "Horse": "mcleish1989", "Pima": "smith1988", "Ecoli": "nakai1991",
            "Sonar": "gorman1988", "German": "hofmann1994", "Glass": "german1987",
            "Ionosphere": "sigillito1989"}
    L = ["\\begin{table}[t]",
         "\\caption{Benchmark datasets. $N$: instances used; $p$: raw input features; "
         "Cat.: categorical (nominal or binary) features among the $p$; $K$: classes; "
         "IR: imbalance ratio = size of the largest class / size of the smallest class; "
         "Maj.: majority-class rate (accuracy of the majority-class baseline); "
         "Miss.: percentage of missing feature values; Year: year the dataset was donated "
         "to the UCI repository. Extended-suite datasets with more than 5{,}000 instances were "
         "stratified-subsampled to 5{,}000 (original size in parentheses).}\\label{tab:datasets}",
         "\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llrrrrrrrr}",
         "\\toprule",
         "Dataset & Domain & $N$ & $p$ & Cat. & $K$ & IR & Maj. & Miss.\\,(\\%) & Year \\\\",
         "\\midrule",
         "\\multicolumn{10}{@{}l}{\\emph{Core suite}} \\\\"]
    for _, r in df.iterrows():
        if r.dataset == "contraceptive_method":
            L.append("\\midrule")
            L.append("\\multicolumn{10}{@{}l}{\\emph{Extended suite (OpenML-CC18 / PMLB)}} \\\\")
        nm = SHORT[r.dataset]
        if r.dataset in cite:
            nm += f"~\\cite{{{cite[r.dataset]}}}"
        n = f"{r.n:,}".replace(",", "{,}")
        if r.n_orig != r.n:
            n += f" ({r.n_orig:,})".replace(",", "{,}")
        L.append(f"{nm} & {r.domain} & {n} & {r.p} & {r.n_cat} & {r.classes} & "
                 f"{r.imbalance:.2f} & {r.majority_rate:.3f} & {r.missing_pct:.1f} & {r.year} \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, "datasets.tex"), "w").write("\n".join(L))
    NUM["datasets"] = df.set_index("dataset").to_dict(orient="index")
    NUM["n_domains_core"] = int(df[df.dataset.isin(data.CORE_ORDER)].domain.nunique())
    NUM["n_domains_all"] = int(df.domain.nunique())
    return df


# =========================================================================
def main_tables(R, tag="core"):
    order = [d for d in data.CORE_ORDER if d in R]
    out = {}
    for key, cap in [("acc", "Accuracy"), ("bacc", "Balanced accuracy"),
                     ("f1", "Macro-F1 (unweighted mean of the per-class F1 scores; "
                            "computed with the same definition for binary and multiclass tasks)"),
                     ("auc", "AUC-ROC (binary: ROC AUC of $P(y{=}1)$; multiclass: macro-averaged "
                             "one-versus-rest AUC)"),
                     ("mcc", "Matthews correlation coefficient")]:
        mu, sd = cached_table(tag, R, order, ["Majority"] + MAIN, key)
        out[key] = (mu, sd)
        latex_metric_table(mu, sd, key, f"tab:{tag}_{key}",
                           f"{cap} on the core suite under Protocol~A (mean$\\pm$standard "
                           f"deviation over the 25 outer folds of $5\\times5$ repeated "
                           f"stratified cross-validation). Bold: best of the nine compared "
                           f"models; the majority-class baseline (Maj.) is shown for "
                           f"reference and is not ranked.")
    return out


def friedman_block(R, order, models, keys=("acc", "bacc", "f1", "auc", "mcc"), tag="core"):
    res = {}
    for key in keys:
        mu, _ = cached_table(tag, R, order, models, key)
        fr = S.friedman(mu.values, higher_better=key not in LOWER)
        res[key] = fr
    return res


def wilcoxon_block(R, order, key="acc", ref="CASE", others=BASEL, tag="core"):
    mu, _ = cached_table(tag, R, order, [ref] + others, key)
    return S.wilcoxon_vs(mu[ref].values, mu[others].values, others,
                         higher_better=key not in LOWER)


def stats_table(fr, wil, wil_b, wil_f, label, N, extra_caption=""):
    L = ["\\begin{table}[t]",
         f"\\caption{{Statistical comparison over the {N} datasets{extra_caption}. "
         "Panel~A: Friedman test with the Iman--Davenport correction~\\cite{iman1980}; "
         "Nemenyi critical difference~\\cite{nemenyi1963,demsar2006}. Panel~B: two-sided "
         "Wilcoxon signed-rank test~\\cite{wilcoxon1945} of CASE against each baseline on the "
         "per-dataset mean scores; $\\Delta$: mean difference (CASE $-$ baseline) with 95\\% "
         "bootstrap confidence interval; W/T/L: datasets on which CASE wins/ties/loses; "
         "$r_{rb}$: matched-pairs rank-biserial correlation (effect size; $+1$ means CASE is better "
         "on every dataset)~\\cite{kerby2014}; $p_{\\text{Holm}}$ and $p_{\\text{Bonf}}$: $p$-values "
         "adjusted for the eight comparisons with Holm's step-down~\\cite{holm1979} and "
         "Bonferroni procedures. Balanced-accuracy and macro-F1 columns report only "
         f"$\\Delta$ and $p_{{\\text{{Holm}}}}$.}}\\label{{{label}}}",
         "\\scriptsize\\setlength{\\tabcolsep}{2.2pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccccccc}",
         "\\toprule",
         "\\multicolumn{8}{@{}l}{\\textbf{Panel A: Friedman test} ($k=9$ models)} \\\\",
         "\\midrule",
         "Metric & $\\chi^2_F$ & $p$ & $F_F$ & $p$ & CD & Best avg.\\ rank & CASE avg.\\ rank \\\\"]
    k = 9
    cd = S.nemenyi_cd(k, N)
    for key in ("acc", "bacc", "f1", "auc", "mcc"):
        f = fr[key]
        ar = f["avg_rank"]
        bi = int(np.argmin(ar))
        L.append(f"{METRIC_NAMES[key]} & {f['chi2']:.2f} & {pfmt(f['p'])} & {f['ff']:.2f} & "
                 f"{pfmt(f['p_ff'])} & {cd:.2f} & {MAIN[bi]} ({ar[bi]:.2f}) & {ar[-1]:.2f} \\\\")
    L += ["\\midrule",
          "\\multicolumn{8}{@{}l}{\\textbf{Panel B: Wilcoxon signed-rank, CASE vs.\\ baseline}} \\\\",
          "\\midrule",
          " & \\multicolumn{5}{c}{Accuracy} & Bal.\\ acc. & Macro-F1 \\\\",
          "\\cmidrule{2-6}\\cmidrule{7-7}\\cmidrule{8-8}",
          "Baseline & $\\Delta$ [95\\% CI] & W/T/L & $r_{rb}$ & $p$ & $p_{\\text{Holm}}$ ($p_{\\text{Bonf}}$) & "
          "$\\Delta$ ($p_{\\text{Holm}}$) & $\\Delta$ ($p_{\\text{Holm}}$) \\\\",
          "\\midrule"]
    for w, wb, wf in zip(wil, wil_b, wil_f):
        L.append(f"{w['model']} & {w['mean_diff']:+.3f} [{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}] & "
                 f"{w['wins']}/{w['ties']}/{w['losses']} & {w['r_rb']:+.2f} & {pfmt(w['p'])} & "
                 f"{pfmt(w['p_holm'])} ({pfmt(w['p_bonf'])}) & {wb['mean_diff']:+.3f} ({pfmt(wb['p_holm'])}) & "
                 f"{wf['mean_diff']:+.3f} ({pfmt(wf['p_holm'])}) \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(L))


def pfmt(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def wil_to_json(w):
    return [{k: (v.tolist() if isinstance(v, np.ndarray) else
                 (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v))
             for k, v in r.items()} for r in w]


def fr_to_json(fr):
    return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else float(vv))
                for kk, vv in v.items()} for k, v in fr.items()}


# =========================================================================
def cd_diagram(avg_rank, names, cd, title, fname, highlight="CASE"):
    k = len(names)
    order = np.argsort(avg_rank)
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(hi + 2.6, lo - 2.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.8, lo, hi, color=INK, lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.8, 0.84, color=INK, lw=1)
        ax.text(t, 0.87, str(t), ha="center", va="bottom", fontsize=8)
    # CD bar
    ax.hlines(0.97, lo, lo + cd, color=INK, lw=1.5)
    ax.vlines([lo, lo + cd], 0.95, 0.99, color=INK, lw=1.5)
    ax.text(lo + cd / 2, 0.99, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    half = (k + 1) // 2
    left = order[:half]
    right = order[half:][::-1]
    for i, j in enumerate(left):
        y = 0.68 - i * 0.12
        c = BLUE if names[j] == highlight else INK
        ax.plot([avg_rank[j], avg_rank[j], lo - 0.05], [0.8, y, y], color=c, lw=1)
        ax.text(lo - 0.15, y, f"{names[j]} ({avg_rank[j]:.2f})", ha="left", va="center",
                fontsize=8, color=c, fontweight="bold" if c == BLUE else None)
    for i, j in enumerate(right):
        y = 0.68 - i * 0.12
        c = BLUE if names[j] == highlight else INK
        ax.plot([avg_rank[j], avg_rank[j], hi + 0.05], [0.8, y, y], color=c, lw=1)
        ax.text(hi + 0.15, y, f"({avg_rank[j]:.2f}) {names[j]}", ha="right", va="center",
                fontsize=8, color=c, fontweight="bold" if c == BLUE else None)
    # cliques: maximal groups with range < cd
    srt = np.sort(avg_rank)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and srt[j + 1] - srt[i] < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    cliques = [c for c in cliques if not any(o[0] <= c[0] and c[1] <= o[1] and o != c for o in cliques)]
    for n, (i, j) in enumerate(cliques):
        y = 0.76 - n * 0.035
        ax.hlines(y, srt[i] - 0.03, srt[j] + 0.03, color=GRAY, lw=3, alpha=0.9)
    ax.set_title(title, fontsize=9, pad=14)
    save(fig, fname)
    return cliques


def rank_heatmap(mu, fname, title):
    R = S.ranks(mu.values)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(mu) + 1.4))
    im = ax.imshow(R, cmap="Blues_r", vmin=1, vmax=len(mu.columns), aspect="auto")
    ax.set_xticks(range(len(mu.columns)), mu.columns, rotation=0, fontsize=8)
    ax.set_yticks(range(len(mu)), [SHORT[d] for d in mu.index], fontsize=8)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:g}", ha="center", va="center", fontsize=7,
                    color="white" if R[i, j] < 3.5 else INK)
    j = list(mu.columns).index("CASE")
    ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, R.shape[0], fill=False, ec=ORANGE, lw=2))
    ax.set_xticks(np.arange(-.5, R.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, R.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Rank (1 = best)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=9)
    save(fig, fname)


def effect_plot(wil_sets, fname):
    """Forest plot: mean accuracy difference CASE - baseline with 95% bootstrap CI."""
    fig, axes = plt.subplots(1, len(wil_sets), figsize=(7.2, 3.6), sharey=True)
    if len(wil_sets) == 1:
        axes = [axes]
    for ax, (title, wil) in zip(axes, wil_sets):
        y = np.arange(len(wil))[::-1]
        for yi, w in zip(y, wil):
            sig = w["p_holm"] < 0.05
            c = BLUE if sig else GRAY
            ax.plot(w["ci"], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.plot(w["mean_diff"], yi, "o", ms=6, color=c, mec="white", mew=1.2)
            ph = "<0.001" if w["p_holm"] < 0.001 else f"={w['p_holm']:.3f}"
            ax.text(0.99, yi + 0.33, f"$r_{{rb}}$={w['r_rb']:+.2f}, $p_{{Holm}}${ph}",
                    fontsize=6.3, color="#52514e", ha="right", va="center",
                    transform=ax.get_yaxis_transform())
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_ylim(-0.6, len(wil) - 0.3)
        lo_, hi_ = ax.get_xlim(); ax.set_xlim(min(lo_, -0.01), hi_ + 0.04 * (hi_ - lo_))
        ax.set_yticks(y, [w["model"] for w in wil])
        ax.set_xlabel("Accuracy difference, CASE $-$ baseline")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    save(fig, fname)


def boxplots(R, order, fname):
    keys = ["acc", "bacc", "f1"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
    for ax, key in zip(axes, keys):
        mu, _ = cached_table("core", R, order, ["Majority"] + MAIN, key)
        vals = [mu[m].values for m in ["Majority"] + MAIN]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=1.2),
                        flierprops=dict(marker="o", ms=3, mfc=GRAY, mec="none"))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(BLUE if i == len(vals) - 1 else ("#e6e5e0" if i == 0 else "#c9d9ee"))
            b.set_edgecolor("#52514e")
        ax.set_xticks(range(1, len(vals) + 1), ["Maj."] + MAIN, rotation=60, fontsize=7)
        ax.set_title(METRIC_NAMES[key], fontsize=9)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    save(fig, fname)


def reliability_fig(R, order, models, fname):
    colors = {"CatBoost": ORANGE, "Stack": GRAY, "CASE": BLUE, "LR": "#1baf7a",
              "CatBoost+Platt": "#6250d6", "kNN": "#e87ba4"}
    n = len(order)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.65 * nrow + 0.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, d in zip(axes, order):
        r = R[d]
        y = r["y"]
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8, ls="--")
        for m in models:
            # pool predictions of all 25 outer folds (each instance appears 5 times)
            Ys = np.concatenate([y[te] for (_, _, _, te) in r["folds"]])
            Ps = np.vstack(r["P"][m])
            rel = M.reliability(Ys, Ps)
            ece = np.mean([M.ece_toplabel(y[te], P) for P, (_, _, _, te) in zip(r["P"][m], r["folds"])])
            ax.plot(rel[:, 0], rel[:, 1], "-o", ms=2.5, lw=1.2, color=colors[m], label=m)
            ax.text(0.98, 0.04 + 0.085 * models.index(m), f"{ece:.3f}", color=colors[m],
                    fontsize=6, ha="right", transform=ax.transAxes)
        ax.set_title(SHORT[d], fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(models), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.supxlabel("Mean top-label confidence", fontsize=8)
    fig.supylabel("Observed accuracy", fontsize=8)
    fig.tight_layout()
    save(fig, fname)


def time_fig(R, order, fname):
    models = MAIN
    med = {m: np.median([np.median(R[d]["T"][m]) for d in order]) for m in models}
    pima = {m: np.median(R["Pima"]["T"][m]) for m in models}
    mu, _ = cached_table("core", R, order, MAIN, "acc")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    srt = sorted(models, key=lambda m: pima[m])
    ax.barh(range(len(srt)), [pima[m] for m in srt],
            color=[BLUE if m == "CASE" else "#9dbfe9" for m in srt], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(range(len(srt)), srt, fontsize=8)
    for i, m in enumerate(srt):
        ax.text(pima[m] * 1.15, i, f"{pima[m]*1000:.0f} ms" if pima[m] < 1 else f"{pima[m]:.2f} s",
                va="center", fontsize=7)
    ax.set_xlabel("Training time per outer fold, Pima (s, log scale)")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax = axes[1]
    for m in models:
        x, yv = med[m], mu[m].mean()
        ax.plot(x, yv, "o", ms=6, color=BLUE if m == "CASE" else "#9dbfe9", mec="white", mew=1)
        ax.annotate(m, (x, yv), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Median training time per fold, all core datasets (s)")
    ax.set_ylabel("Mean accuracy (core suite)")
    ax.grid(color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    save(fig, fname)
    return med, pima


def corr_fig(R, order, fname):
    from pipeline import OOF_MODELS
    C = np.nanmean([np.nanmean(R[d]["oof_corr"], 0) for d in order], 0)
    E = np.nanmean([np.nanmean(R[d]["oof_err_corr"], 0) for d in order], 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, A, t in zip(axes, [C, E], ["OOF probability of the true class",
                                       "OOF error indicators"]):
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), OOF_MODELS, rotation=45, fontsize=7, ha="right")
        ax.set_yticks(range(6), OOF_MODELS, fontsize=7)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if A[i, j] > 0.6 else INK)
        ax.set_title(t, fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Pearson correlation")
    save(fig, fname)
    return C, E


# =========================================================================
def ablation_table(R, order, tag, label, caption, N):
    rows = []
    mus = {k: cached_table(tag, R, order, ABL, k)[0] for k in ("acc", "bacc", "f1", "auc", "ll", "brier", "ece")}
    wil = {k: S.wilcoxon_vs(mus[k]["CASE"].values, mus[k][ABL[1:]].values, ABL[1:],
                            higher_better=k not in LOWER) for k in ("acc", "ece", "ll")}
    Rk = S.ranks(mus["acc"].values).mean(0)
    for i, m in enumerate(ABL):
        row = dict(model=m, rank=Rk[i])
        for k in mus:
            row[k] = mus[k][m].mean()
        if i > 0:
            for k in ("acc", "ece", "ll"):
                row[f"p_{k}"] = wil[k][i - 1]["p_holm"]
        rows.append(row)
    df = pd.DataFrame(rows)
    desc = {"CASE": "4 trees + LR meta + Platt (proposed)",
            "Stack": "4 trees + LR meta, no calibration",
            "CASE-Iso": "4 trees + LR meta + isotonic",
            "SoftVote": "average of the 4 trees' probabilities",
            "CatBoost": "single CatBoost, uncalibrated",
            "CatBoost+Platt": "single CatBoost + Platt",
            "CatBoost+Iso": "single CatBoost + isotonic",
            "SelectBest": "single model with best inner-CV log-loss",
            "CASE-H4": "CatBoost, RF, SVM, LR + LR meta + Platt",
            "CASE-H6": "4 trees + SVM + LR + LR meta + Platt"}
    L = ["\\begin{table}[t]", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\scriptsize\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccccc}",
         "\\toprule",
         "Variant & Rank & Acc. & Bal.\\ acc. & Macro-F1 & Log-loss & Brier & ECE & "
         "$p_{\\text{Holm}}$ (Acc./ECE) \\\\", "\\midrule"]
    for _, r in df.iterrows():
        ps = "--" if r.model == "CASE" else f"{pfmt(r.p_acc)} / {pfmt(r.p_ece)}"
        L.append(f"{r.model} & {r['rank']:.2f} & {r.acc:.4f} & {r.bacc:.4f} & "
                 f"{r.f1:.4f} & {r.ll:.3f} & {r.brier:.3f} & {r.ece:.3f} & {ps} \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(L))
    return df, wil


def calib_table(R, order, label):
    models = ["Majority", "LR", "kNN", "RF", "CatBoost", "CatBoost+Platt", "SoftVote", "Stack", "CASE"]
    mu, sd = cached_table("core", R, order, models, "ece")
    bmu, _ = cached_table("core", R, order, models, "brier")
    L = ["\\begin{table}[t]",
         "\\caption{Expected calibration error (top-label ECE, Eq.~\\ref{eq:ece}, $M=10$ bins; "
         "lower is better) on all ten core datasets, mean over the 25 outer folds. The last two rows "
         "give the mean ECE and mean Brier score over datasets. Bold: lowest ECE among the "
         "non-trivial models (the majority-class baseline is trivially calibrated but "
         "uninformative).}\\label{" + label + "}",
         "\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l" + "c" * len(models) + "}",
         "\\toprule",
         "Dataset & " + " & ".join(["Maj."] + [m.replace("CatBoost+Platt", "CatB.+Platt") for m in models[1:]]) + " \\\\",
         "\\midrule"]
    for d in order:
        row = mu.loc[d, models[1:]].values
        b = row.min()
        cells = [fmt(mu.loc[d, "Majority"])]
        for m in models[1:]:
            v = mu.loc[d, m]
            cells.append(f"\\textbf{{{fmt(v)}}}" if abs(v - b) < 5e-5 else fmt(v))
        L.append(f"{SHORT[d]} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    L.append("Mean ECE & " + " & ".join(fmt(mu[m].mean()) for m in models) + " \\\\")
    L.append("Mean Brier & " + " & ".join(fmt(bmu[m].mean()) for m in models) + " \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(L))
    return mu, bmu


def lodo(R, order):
    mu, _ = cached_table("core", R, order, MAIN, "acc")
    rows = []
    for d in order:
        sub = mu.drop(index=d)
        ar = S.ranks(sub.values).mean(0)
        pos = int(stats_rank_pos(ar, MAIN.index("CASE")))
        w = S.wilcoxon_vs(sub["CASE"].values, sub[["CatBoost", "RF", "LR", "SVM"]].values,
                          ["CatBoost", "RF", "LR", "SVM"])
        rows.append(dict(left_out=d, case_rank=ar[-1], case_pos=pos,
                         best=MAIN[int(np.argmin(ar))], best_rank=ar.min(),
                         d_cat=(sub["CASE"] - sub["CatBoost"]).mean(),
                         d_rf=(sub["CASE"] - sub["RF"]).mean(),
                         d_lr=(sub["CASE"] - sub["LR"]).mean(),
                         p_cat=w[0]["p"], p_rf=w[1]["p"], p_lr=w[2]["p"]))
    df = pd.DataFrame(rows)
    L = ["\\begin{table}[t]",
         "\\caption{Leave-one-dataset-out (LODO) analysis of accuracy on the core suite. Each row "
         "removes one dataset and recomputes the average ranks of the nine models over the "
         "remaining nine datasets and the mean accuracy difference of CASE against CatBoost, RF "
         "and LR (raw Wilcoxon $p$ in parentheses).}\\label{tab:lodo}",
         "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcclccc}",
         "\\toprule",
         "Left out & CASE rank & CASE position & Best model (rank) & $\\Delta$ vs CatBoost & "
         "$\\Delta$ vs RF & $\\Delta$ vs LR \\\\", "\\midrule"]
    for _, r in df.iterrows():
        L.append(f"{SHORT[r.left_out]} & {r.case_rank:.2f} & {r.case_pos} & {r.best} ({r.best_rank:.2f}) & "
                 f"{r.d_cat:+.4f} ({r.p_cat:.3f}) & {r.d_rf:+.4f} ({r.p_rf:.3f}) & {r.d_lr:+.4f} ({r.p_lr:.3f}) \\\\")
    full = mu
    L.append("\\midrule")
    ar = S.ranks(full.values).mean(0)
    L.append(f"None (all 10) & {ar[-1]:.2f} & {int(stats_rank_pos(ar, 8))} & {MAIN[int(np.argmin(ar))]} ({ar.min():.2f}) & "
             f"{(full['CASE']-full['CatBoost']).mean():+.4f} & {(full['CASE']-full['RF']).mean():+.4f} & "
             f"{(full['CASE']-full['LR']).mean():+.4f} \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, "lodo.tex"), "w").write("\n".join(L))
    return df


def stats_rank_pos(ar, i):
    return 1 + int((ar < ar[i] - 1e-12).sum())


def recall_table(R, order, label):
    models = ["Majority", "LR", "RF", "CatBoost", "Stack", "CASE"]
    sets = [d for d in ["Haberman", "German", "Pima", "Ecoli", "Glass"] if d in R]
    L = ["\\begin{table}[t]",
         "\\caption{Per-class recall (mean over the 25 outer folds) on the five core datasets with "
         "the largest imbalance or the lowest balanced accuracy, Protocol~A. Classes are listed from "
         "the largest to the smallest; $n_c$: class size.}\\label{" + label + "}",
         "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llr" + "c" * len(models) + "}",
         "\\toprule",
         "Dataset & Class & $n_c$ & " + " & ".join(["Maj."] + models[1:]) + " \\\\", "\\midrule"]
    names = {"Haberman": ["survived $\\geq$5 y", "died $<$5 y"], "German": ["good", "bad"],
             "Pima": ["negative", "positive"]}
    for d in sets:
        y = R[d]["y"]
        cnt = np.bincount(y)
        rec = {m: np.mean([x["recall"] for x in M.all_fold_metrics(R[d], m)], 0) for m in models}
        for c in np.argsort(-cnt):
            cname = names.get(d, None)
            cname = cname[c] if cname else f"class {c}"
            L.append(f"{SHORT[d] if c == np.argmax(cnt) else ''} & {cname} & {cnt[c]} & " +
                     " & ".join(f"{rec[m][c]:.3f}" for m in models) + " \\\\")
        if d != sets[-1]:
            L.append("\\midrule")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(L))


# =========================================================================
def protocol_b(RA, RB, order):
    rows = []
    for m in ["Majority"] + MAIN + ["Stack"]:
        r = dict(model=m)
        for tag, R in (("A", RA), ("B", RB)):
            for k in ("acc", "bacc", "f1", "ece"):
                mu, _ = cached_table("core" if tag == "A" else "coreB", R, order, ["Majority"] + MAIN + ["Stack"], k)
                r[f"{k}_{tag}"] = mu[m].mean()
        rows.append(r)
    df = pd.DataFrame(rows)
    rk = {}
    for tag, R in (("A", RA), ("B", RB)):
        for k in ("acc", "bacc", "f1"):
            mu, _ = cached_table("core" if tag == "A" else "coreB", R, order, ["Majority"] + MAIN, k)
            rk[(tag, k)] = S.ranks(mu[MAIN].values).mean(0)
    L = ["\\begin{table}[t]",
         "\\caption{Effect of the imbalance-handling protocol (core suite, means over the ten "
         "datasets; average ranks among the nine models in parentheses). Protocol~A: no "
         "imbalance correction for any model. Protocol~B: class-balanced weighting for every "
         "model (\\texttt{class\\_weight}/\\texttt{sample\\_weight}; SMOTE with adaptive $k$ for "
         "kNN; balanced meta-learner for CASE).}\\label{tab:protocols}",
         "\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccccc}",
         "\\toprule",
         " & \\multicolumn{4}{c}{Protocol A (no correction)} & \\multicolumn{4}{c}{Protocol B (balanced)} \\\\",
         "\\cmidrule{2-5}\\cmidrule{6-9}",
         "Model & Acc. & Bal.\\ acc. & Macro-F1 & ECE & Acc. & Bal.\\ acc. & Macro-F1 & ECE \\\\",
         "\\midrule"]
    for _, r in df.iterrows():
        def cell(tag, k):
            v = r[f"{k}_{tag}"]
            if r.model in MAIN and k != "ece":
                return f"{v:.3f} ({rk[(tag, k)][MAIN.index(r.model)]:.1f})"
            return f"{v:.3f}"
        if r.model == "Stack":
            L.append("\\midrule")
        L.append(f"{'Stack (CASE w/o Platt)' if r.model == 'Stack' else r.model} & " +
                 " & ".join(cell(t, k) for t in "AB" for k in ("acc", "bacc", "f1", "ece")) + " \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, "protocols.tex"), "w").write("\n".join(L))
    return df, rk


def hpo_analysis(RA, H, order):
    """Tuned vs default on the paired first 5 folds."""
    order = [d for d in order if d in H]
    models = BASEL + ["CASE", "CASE-H6"]
    rows, tuned_mu, def_mu = [], {}, {}
    for d in order:
        r = H[d]
        for m in models:
            acc_t = np.mean([(P.argmax(1) == r["y"][te]).mean() for P, (_, _, _, te) in zip(r["P"][m], r["folds"])])
            ra = RA[d]
            acc_d = np.mean([(P.argmax(1) == ra["y"][te]).mean()
                             for P, (rep, _, _, te) in zip(ra["P"][m], ra["folds"]) if rep == 0])
            tuned_mu.setdefault(m, []).append(acc_t)
            def_mu.setdefault(m, []).append(acc_d)
    T = pd.DataFrame(tuned_mu, index=order)
    D = pd.DataFrame(def_mu, index=order)
    fr_t = S.friedman(T[MAIN].values)
    wil_t = S.wilcoxon_vs(T["CASE"].values, T[BASEL].values, BASEL)
    # tuned vs default per model
    from scipy.stats import wilcoxon
    for m in models:
        dd = T[m] - D[m]
        nz = dd.abs() > 1e-12
        p = wilcoxon(T[m][nz], D[m][nz]).pvalue if nz.sum() > 0 else 1.0
        sel_default = np.mean([np.mean(np.array(H[d]["best_idx"].get(m, [np.nan])) == 0) for d in order]) \
            if m in BASEL else np.nan
        rows.append(dict(model=m, default=D[m].mean(), tuned=T[m].mean(), diff=dd.mean(),
                         wins=int((dd > 1e-12).sum()), losses=int((dd < -1e-12).sum()), p=p,
                         rank_t=fr_t["avg_rank"][MAIN.index(m)] if m in MAIN else np.nan,
                         frac_default=sel_default))
    df = pd.DataFrame(rows)
    L = ["\\begin{table}[t]",
         "\\caption{Nested-CV hyper-parameter optimisation (core suite, Protocol~A, outer 5-fold "
         "split of repeat~1, paired with the default-configuration results on the same folds). "
         "Inner loop: random search over 10 configurations (the default configuration of "
         "Table~\\ref{tab:hyperparams} plus nine samples from the space in Table~\\ref{tab:hpspace}) "
         "scored by 3-fold log-loss on the outer training fold only. Def./Tuned: mean accuracy over the "
         "ten datasets; W/L: datasets on which tuning helps/hurts; $p$: Wilcoxon tuned vs.\\ default; "
         "Rank: average accuracy rank among the nine tuned models; Def.\\ sel.: fraction of outer folds "
         "in which the default configuration was selected. CASE and CASE-H6 are built from the tuned "
         "base models of each outer fold.}\\label{tab:hpo}",
         "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccccccc}",
         "\\toprule", "Model & Def. & Tuned & $\\Delta$ & W/L & $p$ & Rank (tuned) & Def.\\ sel. \\\\",
         "\\midrule"]
    for _, r in df.iterrows():
        L.append(f"{r.model} & {r.default:.4f} & {r.tuned:.4f} & {r['diff']:+.4f} & {r.wins}/{r.losses} & "
                 f"{r.p:.3f} & {'--' if not np.isfinite(r.rank_t) else f'{r.rank_t:.2f}'} & "
                 f"{'--' if not np.isfinite(r.frac_default) else f'{r.frac_default:.2f}'} \\\\")
    L += ["\\midrule",
          f"\\multicolumn{{8}}{{@{{}}l}}{{Friedman (tuned, $k=9$): $\\chi^2_F={fr_t['chi2']:.2f}$, "
          f"$p={fr_t['p']:.3f}$; Iman--Davenport $F_F={fr_t['ff']:.2f}$, $p={fr_t['p_ff']:.3f}$.}} \\\\",
          "\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, "hpo.tex"), "w").write("\n".join(L))
    return df, fr_t, wil_t, T, D


def automl_analysis(RA, AM, order):
    """CASE vs AutoGluon / TabPFN on the paired folds (first 2 repeats)."""
    rows = []
    out = {}
    for meth, R in AM.items():
        ds = [d for d in order if d in R]
        acc = {"CASE": [], "CASE-H6": [], "CatBoost": [], meth: []}
        ece = {"CASE": [], "CASE-H6": [], "CatBoost": [], meth: []}
        ll = {"CASE": [], "CASE-H6": [], "CatBoost": [], meth: []}
        tt = {"CASE": [], meth: []}
        for d in ds:
            r = R[d]
            nf = len(r["folds"])
            ra = RA[d]
            for m in ["CASE", "CASE-H6", "CatBoost", meth]:
                src = r if m == meth else ra
                fm = [M.fold_metrics(src["y"][te], P) for P, (_, _, _, te) in
                      list(zip(src["P"][m], src["folds"]))[:nf]]
                acc[m].append(np.mean([x["acc"] for x in fm]))
                ece[m].append(np.mean([x["ece"] for x in fm]))
                ll[m].append(np.mean([x["ll"] for x in fm]))
            tt["CASE"].append(np.median(ra["T"]["CASE"][:nf]))
            tt[meth].append(np.median(r["T"][meth]))
        A = pd.DataFrame(acc, index=ds)
        from scipy.stats import wilcoxon
        w = wilcoxon(A["CASE"], A[meth]) if (A["CASE"] - A[meth]).abs().sum() > 0 else None
        out[meth] = dict(acc=A, ece=pd.DataFrame(ece, index=ds), ll=pd.DataFrame(ll, index=ds),
                         p=w.pvalue if w else 1.0, r_rb=S.rank_biserial(A["CASE"], A[meth]),
                         t_case=np.median(tt["CASE"]), t_meth=np.median(tt[meth]))
    return out


# =========================================================================
def main():
    NUM.clear()
    ds = dataset_table()
    RA = load("core_A_*.pkl")
    RB = load("core_B_*.pkl")
    order = [d for d in data.CORE_ORDER if d in RA]
    NUM["order"] = order
    tabs = main_tables(RA)
    fr = friedman_block(RA, order, MAIN)
    wil = {k: wilcoxon_block(RA, order, k) for k in ("acc", "bacc", "f1", "auc", "mcc", "ece", "ll")}
    stats_table(fr, wil["acc"], wil["bacc"], wil["f1"], "tab:stats", len(order))
    cd = S.nemenyi_cd(9, len(order))
    NUM["core"] = dict(friedman=fr_to_json(fr), wilcoxon={k: wil_to_json(v) for k, v in wil.items()},
                       cd=cd)
    for key in ("acc", "bacc", "f1"):
        mu = tabs[key][0][MAIN]
        ar = S.ranks(mu.values).mean(0)
        NUM["core"][f"avg_rank_{key}"] = dict(zip(MAIN, ar.tolist()))
        NUM["core"][f"mean_{key}"] = tabs[key][0].mean().to_dict()
        NUM["core"][f"per_dataset_{key}"] = tabs[key][0].to_dict()
        NUM["core"][f"ranks_{key}"] = pd.DataFrame(S.ranks(mu.values), index=order, columns=MAIN).to_dict()
        # Nemenyi significant pairs
        pairs = [(MAIN[i], MAIN[j], float(abs(ar[i] - ar[j]))) for i in range(9) for j in range(i + 1, 9)
                 if abs(ar[i] - ar[j]) > cd]
        NUM["core"][f"nemenyi_sig_{key}"] = pairs
    NUM["core"]["per_dataset_auc"] = tabs["auc"][0].to_dict()
    NUM["core"]["per_dataset_mcc"] = tabs["mcc"][0].to_dict()
    rank_heatmap(tabs["acc"][0][MAIN], "fig_rank_heatmap",
                 "Per-dataset accuracy rank (core suite, Protocol A)")
    cl = cd_diagram(np.array(list(NUM["core"]["avg_rank_acc"].values())), MAIN, cd,
                    f"Accuracy, core suite (N = {len(order)} datasets, k = 9)", "fig_cd_core")
    cd_diagram(np.array(list(NUM["core"]["avg_rank_bacc"].values())), MAIN, cd,
               f"Balanced accuracy, core suite (N = {len(order)}, k = 9)", "fig_cd_core_bacc")
    boxplots(RA, order, "fig_boxplots")
    calmu, calb = calib_table(RA, order, "tab:ece")
    NUM["ece"] = calmu.to_dict(); NUM["brier"] = calb.to_dict()
    NUM["ece_mean"] = calmu.mean().to_dict(); NUM["brier_mean"] = calb.mean().to_dict()
    reliability_fig(RA, order, ["CatBoost", "Stack", "CASE", "LR"], "fig_calibration")
    med, pima = time_fig(RA, order, "fig_training_time")
    NUM["time_median"] = med; NUM["time_pima"] = pima
    C, E = corr_fig(RA, order, "fig_oof_corr")
    from pipeline import OOF_MODELS
    NUM["oof_corr"] = pd.DataFrame(C, index=OOF_MODELS, columns=OOF_MODELS).to_dict()
    NUM["oof_err_corr"] = pd.DataFrame(E, index=OOF_MODELS, columns=OOF_MODELS).to_dict()
    NUM["oof_corr_per_dataset"] = {d: pd.DataFrame(np.nanmean(RA[d]["oof_corr"], 0), index=OOF_MODELS,
                                                   columns=OOF_MODELS).to_dict() for d in order}
    abl, abl_w = ablation_table(RA, order, "core", "tab:ablation",
                                "Ablation and alternative designs on the core suite (variants defined in Section~\\ref{subsec:variants}; Protocol~A, means over "
                                "ten datasets of the 25-fold means; Rank: average accuracy rank among the ten "
                                "variants). $p_{\\text{Holm}}$: Wilcoxon signed-rank test of each variant "
                                "against CASE (accuracy / ECE), Holm-adjusted over the nine comparisons. All "
                                "variants share the same base-model fits within each fold.", len(order))
    NUM["ablation"] = abl.to_dict(orient="records")
    NUM["ablation_wil"] = {k: wil_to_json(v) for k, v in abl_w.items()}
    NUM["ablation_per_dataset_acc"] = cached_table("core", RA, order, ABL, "acc")[0].to_dict()
    NUM["ablation_per_dataset_ece"] = cached_table("core", RA, order, ABL, "ece")[0].to_dict()
    NUM["select_counts"] = {d: pd.Series(RA[d]["select"]).value_counts().to_dict() for d in order}
    lo = lodo(RA, order)
    NUM["lodo"] = lo.to_dict(orient="records")
    recall_table(RA, order, "tab:recall")
    NUM["recall"] = {d: {m: np.mean([x["recall"] for x in M.all_fold_metrics(RA[d], m)], 0).tolist()
                         for m in ["Majority"] + MAIN + ["Stack"]} for d in order}
    if RB:
        orderB = [d for d in order if d in RB]
        dfb, rk = protocol_b(RA, RB, orderB)
        NUM["protocolB"] = dfb.to_dict(orient="records")
        NUM["protocolB_ranks"] = {f"{a}_{b}": dict(zip(MAIN, v.tolist())) for (a, b), v in rk.items()}
        frB = friedman_block(RB, orderB, MAIN, tag="coreB")
        NUM["protocolB_friedman"] = fr_to_json(frB)
        NUM["protocolB_wil_bacc"] = wil_to_json(wilcoxon_block(RB, orderB, "bacc", tag="coreB"))
        NUM["protocolB_wil_acc"] = wil_to_json(wilcoxon_block(RB, orderB, "acc", tag="coreB"))
        NUM["protocolB_per_dataset"] = {k: cached_table("coreB", RB, orderB, ["Majority"] + MAIN + ["Stack"], k)[0].to_dict()
                                        for k in ("acc", "bacc", "f1")}
    H = load("hpo_A_*.pkl")
    if H:
        df, fr_t, wil_t, T, D = hpo_analysis(RA, H, order)
        NUM["hpo"] = df.to_dict(orient="records")
        NUM["hpo_friedman"] = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in fr_t.items()}
        NUM["hpo_wil"] = wil_to_json(wil_t)
        NUM["hpo_n"] = len(T)
        NUM["hpo_best"] = {d: {m: H[d]["best"][m] for m in BASEL} for d in H}
    AM = {}
    for meth in ("AutoGluon", "TabPFN"):
        R = load(f"automl_{meth}_*.pkl")
        if R:
            AM[meth] = R
    if AM:
        am = automl_analysis(RA, AM, order)
        NUM["automl"] = {m: dict(acc=v["acc"].to_dict(), ece=v["ece"].to_dict(), ll=v["ll"].to_dict(),
                                 mean_acc=v["acc"].mean().to_dict(), mean_ece=v["ece"].mean().to_dict(),
                                 mean_ll=v["ll"].mean().to_dict(), p=float(v["p"]), r_rb=float(v["r_rb"]),
                                 t_case=float(v["t_case"]), t_meth=float(v["t_meth"]), n=len(v["acc"]))
                         for m, v in am.items()}
        automl_table(am)
    E = load("ext_A_*.pkl")
    if E:
        ext_analysis(RA, E, order)
    effect_sets = [("Core suite (N = 10)", wil["acc"])]
    if "combined_wil" in NUM:
        effect_sets.append((f"Core + extended (N = {NUM['combined_N']})", NUM["_combined_wil_raw"]))
    effect_plot(effect_sets, "fig_effects")
    NUM.pop("_combined_wil_raw", None)
    json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
    print("analysis written")


def automl_table(am):
    L = ["\\begin{table}[t]",
         "\\caption{Comparison with automated / foundation-model approaches on the core suite "
         "(Protocol~A; paired outer folds of the first two repeats, i.e.\\ $5\\times2$ = 10 folds). "
         "AutoGluon-Tabular~\\cite{erickson2020}: \\texttt{medium\\_quality} preset, 30\\,s time "
         "limit per fold. TabPFN~\\cite{hollmann2023}: v1 checkpoint, 16 ensemble members. "
         "Values are means over datasets of per-dataset means; $p$: Wilcoxon signed-rank test "
         "of CASE vs.\\ the method on accuracy; Time: median wall-clock time per outer fold "
         "(training and prediction).}\\label{tab:automl}",
         "\\scriptsize",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccccc}",
         "\\toprule", "Method & Acc. & ECE & Log-loss & $p$ (CASE vs.) & Time (s) \\\\", "\\midrule"]
    first = list(am.values())[0]
    L.append(f"CASE & {first['acc']['CASE'].mean():.4f} & {first['ece']['CASE'].mean():.3f} & "
             f"{first['ll']['CASE'].mean():.3f} & -- & {first['t_case']:.2f} \\\\")
    L.append(f"CASE-H6 & {first['acc']['CASE-H6'].mean():.4f} & {first['ece']['CASE-H6'].mean():.3f} & "
             f"{first['ll']['CASE-H6'].mean():.3f} & -- & -- \\\\")
    L.append(f"CatBoost & {first['acc']['CatBoost'].mean():.4f} & {first['ece']['CatBoost'].mean():.3f} & "
             f"{first['ll']['CatBoost'].mean():.3f} & -- & -- \\\\")
    for m, v in am.items():
        L.append(f"{m} & {v['acc'][m].mean():.4f} & {v['ece'][m].mean():.3f} & {v['ll'][m].mean():.3f} & "
                 f"{v['p']:.3f} & {v['t_meth']:.2f} \\\\")
    L += ["\\botrule", "\\end{tabular*}", "\\end{table}"]
    open(os.path.join(TAB, "automl.tex"), "w").write("\n".join(L))


def ext_analysis(RA, E, order):
    eorder = [d for d in data.EXT_ORDER if d in E]
    for key, cap in [("acc", "Accuracy"), ("bacc", "Balanced accuracy"), ("f1", "Macro-F1")]:
        mu, sd = cached_table("ext", E, eorder, ["Majority"] + MAIN, key)
        latex_metric_table(mu, sd, key, f"tab:ext_{key}",
                           f"{cap} on the extended suite (Protocol~A, default hyper-parameters, "
                           f"mean$\\pm$standard deviation over the 10 outer folds of $5\\times2$ repeated stratified cross-validation). "
                           f"Bold: best of the nine compared models; Maj.: majority-class baseline.")
        NUM.setdefault("ext", {})[f"per_dataset_{key}"] = mu.to_dict()
        NUM["ext"][f"mean_{key}"] = mu.mean().to_dict()
        NUM["ext"][f"avg_rank_{key}"] = dict(zip(MAIN, S.ranks(mu[MAIN].values).mean(0).tolist()))
    frE = friedman_block(E, eorder, MAIN, tag="ext")
    NUM["ext"]["friedman"] = fr_to_json(frE)
    NUM["ext"]["N"] = len(eorder)
    NUM["ext"]["wil_acc"] = wil_to_json(wilcoxon_block(E, eorder, "acc", tag="ext"))
    # ablation on extended
    abl, abl_w = ablation_table(E, eorder, "ext", "tab:ablation_ext",
                                "Ablation and alternative designs on the extended suite (Protocol~A; "
                                "same layout as Table~\\ref{tab:ablation}).", len(eorder))
    NUM["ext"]["ablation"] = abl.to_dict(orient="records")
    NUM["ext"]["ablation_wil"] = {k: wil_to_json(v) for k, v in abl_w.items()}
    NUM["ext"]["ece_mean"] = cached_table("ext", E, eorder, ABL + ["LR", "Majority"], "ece")[0].mean().to_dict()
    # combined
    comb = {}
    for key in ("acc", "bacc", "f1", "auc", "mcc"):
        a, _ = cached_table("core", RA, order, MAIN, key)
        b, _ = cached_table("ext", E, eorder, MAIN, key)
        comb[key] = pd.concat([a, b])
    N = len(comb["acc"])
    frC = {k: S.friedman(v.values, higher_better=True) for k, v in comb.items()}
    wC = {k: S.wilcoxon_vs(comb[k]["CASE"].values, comb[k][BASEL].values, BASEL)
          for k in ("acc", "bacc", "f1")}
    NUM["combined_N"] = N
    NUM["combined_friedman"] = fr_to_json(frC)
    NUM["combined_wil"] = {k: wil_to_json(v) for k, v in wC.items()}
    NUM["_combined_wil_raw"] = wC["acc"]
    cdC = S.nemenyi_cd(9, N)
    NUM["combined_cd"] = cdC
    for key in ("acc", "bacc"):
        ar = S.ranks(comb[key].values).mean(0)
        NUM[f"combined_avg_rank_{key}"] = dict(zip(MAIN, ar.tolist()))
        NUM[f"combined_nemenyi_sig_{key}"] = [(MAIN[i], MAIN[j], float(abs(ar[i] - ar[j])))
                                             for i in range(9) for j in range(i + 1, 9)
                                             if abs(ar[i] - ar[j]) > cdC]
    # meta-analysis of dataset characteristics (tested replacement of former Sec. 6.5)
    from scipy.stats import spearmanr, mannwhitneyu
    A = comb["acc"]
    dsi = pd.DataFrame([data.describe(d) for d in A.index]).set_index("dataset")
    best_single = A[BASEL].max(1)
    feats = {"log10 N": np.log10(dsi.n), "p": dsi.p, "K": dsi.classes,
             "log10 IR": np.log10(dsi.imbalance), "cat. fraction": dsi.n_cat / dsi.p}
    trees_ = A[["RF", "XGBoost", "LightGBM", "CatBoost"]].max(1)
    lin = A[["LR", "SVM"]].max(1)
    targets = {"CASE - best single": A["CASE"] - best_single,
               "best tree - best LR/SVM": trees_ - lin}
    abl_acc = pd.concat([cached_table("core", RA, order, ABL, "acc")[0],
                         cached_table("ext", E, eorder, ABL, "acc")[0]])
    targets["CASE-H6 - CASE"] = abl_acc["CASE-H6"] - abl_acc["CASE"]
    meta = []
    for tn, tv in targets.items():
        for fn_, fv in feats.items():
            rho, pv = spearmanr(fv.values, tv.values)
            meta.append(dict(target=tn, feature=fn_, rho=rho, p=pv))
    NUM["meta_analysis"] = meta
    rkA = pd.DataFrame(S.ranks(A[MAIN].values), index=A.index, columns=MAIN)
    isbin = dsi.classes == 2
    u = mannwhitneyu(rkA["CASE"][isbin], rkA["CASE"][~isbin])
    NUM["case_rank_binary"] = float(rkA["CASE"][isbin].mean())
    NUM["case_rank_multi"] = float(rkA["CASE"][~isbin].mean())
    NUM["case_rank_bin_vs_multi_p"] = float(u.pvalue)
    NUM["n_bin"] = int(isbin.sum()); NUM["n_multi"] = int((~isbin).sum())
    NUM["combined_case_ranks"] = rkA["CASE"].to_dict()
    NUM["combined_ablation_mean"] = abl_acc.mean().to_dict()
    wH = S.wilcoxon_vs(abl_acc["CASE-H6"].values, abl_acc[["CASE", "Stack", "SoftVote", "CatBoost+Platt", "SelectBest"]].values,
                       ["CASE", "Stack", "SoftVote", "CatBoost+Platt", "SelectBest"])
    NUM["combined_h6_wil"] = wil_to_json(wH)
    wC2 = S.wilcoxon_vs(abl_acc["CASE"].values, abl_acc[["Stack", "SoftVote", "CatBoost+Platt", "SelectBest", "CASE-H4", "CASE-H6"]].values,
                        ["Stack", "SoftVote", "CatBoost+Platt", "SelectBest", "CASE-H4", "CASE-H6"])
    NUM["combined_case_abl_wil"] = wil_to_json(wC2)
    cd_diagram(np.array(list(NUM["combined_avg_rank_acc"].values())), MAIN, cdC,
               f"Accuracy, core + extended suites (N = {N}, k = 9)", "fig_cd_combined")
    stats_table(frC, wC["acc"], wC["bacc"], wC["f1"], "tab:stats_combined", N,
                " of the core and extended suites")
    # extended reliability for a few
    rank_heatmap(cached_table("ext", E, eorder, MAIN, "acc")[0], "fig_rank_heatmap_ext",
                 "Per-dataset accuracy rank (extended suite, Protocol A)")


if __name__ == "__main__":
    main()
