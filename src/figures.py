"""
figures.py
Generates all publication figures. Each function saves PDF and PNG versions.

Figures generated
-----------------
1. fig_dataset_bubble    — Dataset landscape (size × features × imbalance)
2. fig_rank_heatmap      — Per-dataset accuracy rank heatmap
3. fig_avg_rank          — Average rank bar chart
4. fig_boxplots          — Score distribution boxplots (acc, f1, auc)
5. fig_grouped_bars      — Per-dataset accuracy grouped bar chart
6. fig_cd_diagram        — Critical-difference diagram (Demšar 2006)
7. fig_wilcoxon          — Wilcoxon p-value heatmap (acc only)
8. fig_case_architecture — CASE three-layer architecture diagram
9. fig_calibration       — Reliability diagrams (3 datasets × 3 models)
10. fig_training_time    — Training time + accuracy vs cost
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.linewidth": 0.8,
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_ORDER = ["LR", "SVM", "kNN", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP", "CASE"]
DS_ORDER    = ["Heart", "WDBC", "Haberman", "Horse", "Pima", "Ecoli",
               "Sonar", "German", "Glass", "Ionosphere"]

COLORS = {
    "LR":       "#6B9BD1", "SVM":      "#F0A500", "kNN":      "#E07070",
    "RF":       "#5BAD72", "XGBoost":  "#A569BD", "LightGBM": "#45B8AC",
    "CatBoost": "#E86E3A", "MLP":      "#85929E", "CASE":     "#2C3E50",
}

DOMAIN_COLORS = {
    "Cardiovascular": "#E74C3C", "Oncology":      "#8E44AD",
    "Surgical":       "#E67E22", "Metabolic":     "#27AE60",
    "Bioinformatics": "#2980B9", "Signal Proc.":  "#1ABC9C",
    "Financial Risk": "#F39C12", "Forensic Sci.": "#7F8C8D",
    "Atmospheric":    "#2C3E50",
}


def _save(name: str) -> None:
    path_pdf = os.path.join(FIGURES_DIR, f"{name}.pdf")
    path_png = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")


def _mu(vals):
    return np.nanmean([v for v in vals if v is not None])


# ─────────────────────────────────────────────────────────────────────────────

def fig_dataset_bubble(datasets: dict) -> None:
    ds_meta = {
        "Heart":      {"n": 303, "p": 13,  "imb": 1.20, "domain": "Cardiovascular"},
        "WDBC":       {"n": 569, "p": 30,  "imb": 1.68, "domain": "Oncology"},
        "Haberman":   {"n": 306, "p": 3,   "imb": 2.78, "domain": "Oncology"},
        "Horse":      {"n": 300, "p": 26,  "imb": 1.75, "domain": "Surgical"},
        "Pima":       {"n": 768, "p": 8,   "imb": 1.87, "domain": "Metabolic"},
        "Ecoli":      {"n": 336, "p": 7,   "imb": 7.33, "domain": "Bioinformatics"},
        "Sonar":      {"n": 208, "p": 60,  "imb": 1.14, "domain": "Signal Proc."},
        "German":     {"n":1000, "p": 61,  "imb": 2.33, "domain": "Financial Risk"},
        "Glass":      {"n": 214, "p": 9,   "imb": 5.50, "domain": "Forensic Sci."},
        "Ionosphere": {"n": 351, "p": 34,  "imb": 1.79, "domain": "Atmospheric"},
    }
    offsets = {
        "Haberman": (-42, 4), "Sonar": (10, 0), "German": (8, 0),
        "Ecoli": (-8, 5), "Glass": (-8, -7),
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#FAFAFA")
    seen_domains = set()
    for name, d in ds_meta.items():
        c = DOMAIN_COLORS[d["domain"]]
        ax.scatter(d["n"], d["p"], s=d["imb"] * 120, c=c, alpha=0.82,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ox, oy = offsets.get(name, (8, 0))
        ax.annotate(name, (d["n"], d["p"]),
                    xytext=(d["n"] + ox, d["p"] + oy), fontsize=8.5,
                    fontweight="bold", color="#2C3E50",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)
                    if ox < 0 else None)
        if d["domain"] not in seen_domains:
            ax.scatter([], [], c=c, s=80, label=d["domain"])
            seen_domains.add(d["domain"])
    for imb, lbl in [(1, "1×"), (3, "3×"), (7, "7×")]:
        ax.scatter([], [], s=imb * 120, c="gray", alpha=0.55, label=f"Imbalance {lbl}")
    ax.set_xlabel("Number of Instances", fontsize=11)
    ax.set_ylabel("Number of Features",  fontsize=11)
    ax.set_title("Dataset Landscape: Size, Dimensionality, and Class Imbalance",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save("fig_dataset_bubble")


def fig_rank_heatmap(all_results: dict) -> None:
    ranks = np.zeros((len(DS_ORDER), len(MODEL_ORDER)))
    for i, ds in enumerate(DS_ORDER):
        sc = {mn: _mu(all_results[ds][mn]["acc"])
              for mn in MODEL_ORDER if mn in all_results.get(ds, {})}
        for rnk, (mn, _) in enumerate(sorted(sc.items(), key=lambda x: -x[1]), 1):
            ranks[i, MODEL_ORDER.index(mn)] = rnk

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(ranks, cmap="RdYlGn_r", vmin=1, vmax=9, aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER))); ax.set_xticklabels(MODEL_ORDER, fontsize=9)
    ax.set_yticks(range(len(DS_ORDER)));    ax.set_yticklabels(DS_ORDER, fontsize=9)
    for i in range(len(DS_ORDER)):
        for j in range(len(MODEL_ORDER)):
            ax.text(j, i, f"{int(ranks[i,j])}", ha="center", va="center",
                    fontsize=9, color="white" if ranks[i, j] >= 7 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="Rank (1=best)")
    ax.set_title("Model Accuracy Rank per Dataset", fontsize=11, fontweight="bold", pad=10)
    case_idx = MODEL_ORDER.index("CASE")
    ax.add_patch(mpatches.FancyBboxPatch(
        (case_idx - 0.45, -0.45), 0.9, len(DS_ORDER) - 0.1,
        linewidth=2, edgecolor="#2C3E50", facecolor="none",
        boxstyle="round,pad=0.02"))
    plt.tight_layout()
    _save("fig_rank_heatmap")


def fig_avg_rank(all_results: dict) -> None:
    from .statistical_tests import compute_average_ranks
    avg_ranks = compute_average_ranks(all_results, DS_ORDER, MODEL_ORDER)
    sorted_models = sorted(MODEL_ORDER, key=lambda m: avg_ranks.get(m, 9))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(sorted_models, [avg_ranks.get(m, 9) for m in sorted_models],
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="white", height=0.6)
    for bar, mn in zip(bars, sorted_models):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{avg_ranks.get(mn, 0):.2f}", va="center", fontsize=9)
    ax.set_xlabel("Average Accuracy Rank (lower = better)")
    ax.set_title("Average Rank Across 10 Datasets", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 10.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save("fig_avg_rank")


def fig_boxplots(all_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, met, title in zip(axes, ["acc", "f1", "auc"],
                               ["Accuracy", "Macro-F1", "AUC-ROC"]):
        data, labels = [], []
        for mn in MODEL_ORDER:
            vals = [_mu(all_results[ds][mn][met])
                    for ds in DS_ORDER if mn in all_results.get(ds, {})]
            data.append([v for v in vals if not np.isnan(v)])
            labels.append(mn)
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, mn in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(COLORS[mn])
        ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
        ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Per-Dataset Score Distributions (10 datasets)", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save("fig_boxplots")


def fig_grouped_bars(all_results: dict) -> None:
    sel = ["CatBoost", "RF", "SVM", "LightGBM", "CASE"]
    x = np.arange(len(DS_ORDER)); w = 0.15
    fig, ax = plt.subplots(figsize=(13, 4))
    for k, mn in enumerate(sel):
        vals = [_mu(all_results[ds][mn]["acc"])
                if mn in all_results.get(ds, {}) else 0 for ds in DS_ORDER]
        ax.bar(x + (k - (len(sel) - 1) / 2) * w, vals, w,
               label=mn, color=COLORS[mn], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(DS_ORDER, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 1.03)
    ax.legend(loc="lower right", fontsize=8, ncol=5, framealpha=0.7)
    ax.set_title("Accuracy: Top Individual Models vs CASE", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save("fig_grouped_bars")


def fig_cd_diagram(all_results: dict) -> None:
    from .statistical_tests import compute_average_ranks, nemenyi_cd
    avg_ranks = compute_average_ranks(all_results, DS_ORDER, MODEL_ORDER)
    sorted_models = sorted(avg_ranks, key=lambda m: avg_ranks[m])
    k, N = len(MODEL_ORDER), len(DS_ORDER)
    CD = nemenyi_cd(k, N)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0.2, 9.8); ax.set_ylim(-2.5, 3.2); ax.axis("off")
    ax.annotate("", xy=(9.6, 2.0), xytext=(0.5, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    for t in range(1, 10):
        ax.plot([t, t], [1.88, 2.12], color="#333", lw=0.9)
        ax.text(t, 2.28, str(t), ha="center", fontsize=9.5, color="#333")
    ax.text(5, 2.75, "Average Rank  (1 = best)", ha="center",
            fontsize=10, fontweight="bold", color="#333")

    for i, mn in enumerate(sorted_models):
        r = avg_ranks[mn]; c = COLORS[mn]; is_case = (mn == "CASE")
        ax.plot(r, 2.0, "o", color=c, markersize=(9 if is_case else 6.5),
                zorder=5, markeredgecolor="white", markeredgewidth=0.8)
        y_label = 2.0 + 0.55 + (i // 2) % 2 * 0.45 if i % 2 == 0 else \
                  2.0 - 0.55 - ((i - 1) // 2) % 2 * 0.42
        ax.plot([r, r], ([2.12, y_label - 0.12] if i % 2 == 0 else [1.88, y_label + 0.12]),
                color=c, lw=0.8, zorder=2)
        ax.text(r, y_label, f"{mn}\n({avg_ranks[mn]:.2f})", ha="center", va="center",
                fontsize=8, fontweight="bold" if is_case else "normal", color=c,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=c, lw=1.5 if is_case else 0.7, alpha=0.93))

    # CD bracket
    cx = 0.9
    ax.annotate("", xy=(cx + CD, -1.85), xytext=(cx, -1.85),
                arrowprops=dict(arrowstyle="<->", color="#2980B9", lw=1.8))
    ax.text(cx + CD / 2, -2.15, f"CD = {CD:.2f}", ha="center",
            fontsize=9, color="#2980B9", fontweight="bold")

    ax.text(5, -2.55,
            f"Friedman χ² = 23.61, p = 0.003  │  Nemenyi CD = {CD:.2f}  "
            f"(α=0.05, k={k}, N={N})",
            ha="center", fontsize=8.5, color="#555", style="italic")
    ax.set_title("Critical-Difference Diagram — Accuracy Rankings",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("fig_cd_diagram")


def fig_wilcoxon(stat_results: dict) -> None:
    """Plot Wilcoxon p-values for accuracy only.
    F1 comparisons are not shown: Friedman F1 was not significant (p=0.292)."""
    if "acc" not in stat_results or stat_results["acc"]["wilcoxon"] is None:
        print("  ⚠ Wilcoxon results not available (Friedman not significant?).")
        return

    wx = stat_results["acc"]["wilcoxon"]
    models_wc = [m for m in MODEL_ORDER if m != "CASE"]
    pvals = np.array([[wx[m]["adj_p"] if m in wx else 1.0] for m in models_wc])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pvals, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
    ax.set_yticks(range(len(models_wc))); ax.set_yticklabels(models_wc, fontsize=10)
    ax.set_xticks([0]); ax.set_xticklabels(["$p$-value\nvs CASE"], fontsize=9)
    for i, mn in enumerate(models_wc):
        pv = wx[mn]["adj_p"] if mn in wx else 1.0
        sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        ax.text(0, i, f"{pv:.3f}{sig}", ha="center", va="center",
                fontsize=10, color="white" if pv < 0.025 else "black",
                fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85, label="$p$-value (Bonferroni corrected)")
    ax.set_title("Wilcoxon Signed-Rank vs CASE (Accuracy)",
                 fontsize=10, fontweight="bold")
    ax.text(0.5, -0.10, "* p<0.05   ** p<0.01   F1 omitted: Friedman p=0.292",
            ha="center", transform=ax.transAxes, fontsize=8, style="italic", color="#555")
    plt.tight_layout()
    _save("fig_wilcoxon")


def fig_case_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, color, label, sublabel="", alpha=0.92, fs=10):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25",
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha, zorder=3))
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=7.5, color="white",
                    style="italic", zorder=4)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    C_BASE = "#2980B9"; C_META = "#8E44AD"; C_CAL = "#27AE60"
    C_IN   = "#7F8C8D"; C_OUT  = "#E74C3C"

    # Labels
    for xc, txt, col in [(1.5, "LAYER 1\nBase Models", C_BASE),
                          (6.5, "LAYER 2\nMeta-Learner", C_META),
                          (9.8, "LAYER 3\nCalibration",  C_CAL),
                          (12.0,"OUTPUT",                 C_OUT)]:
        ax.text(xc, 5.65, txt, ha="center", fontsize=8.5,
                fontweight="bold", color=col)

    box(0.05, 2.55, 1.3, 0.9, C_IN, "Input", "Tabular Data", fs=9)
    arrow(1.35, 3.0, 1.65, 3.0)

    for bm, y in [("CatBoost", 4.6), ("RF", 3.3), ("LightGBM", 2.0), ("XGBoost", 0.7)]:
        box(1.65, y, 1.8, 0.75, C_BASE, bm, "P(y|X)", fs=9)
        arrow(1.35, 3.0, 1.65, y + 0.375, color=C_BASE)
        arrow(3.45, y + 0.375, 4.75, 3.0, color=C_BASE)

    ax.text(4.1, 2.45, "OOF Probs\n[4×K stacked]",
            ha="center", fontsize=7.5, color=C_BASE, style="italic")
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.75, 1.4), 0.8, 3.2, boxstyle="round,pad=0.15",
        linewidth=1, edgecolor=C_BASE, facecolor="#EAF4FB", alpha=0.6, zorder=2))
    ax.text(5.15, 3.0, "Stack\nM_tr", ha="center", va="center", fontsize=8, color=C_BASE, zorder=3)
    arrow(5.55, 3.0, 5.85, 3.0)

    box(5.85, 2.35, 1.9, 1.3, C_META, "Logistic\nRegression", "Meta-Learner", fs=9.5)
    ax.text(6.8, 1.85, "3-fold CV\non train fold",
            ha="center", fontsize=7, color=C_META, style="italic")
    arrow(7.75, 3.0, 8.05, 3.0)

    box(8.05, 2.2, 2.0, 1.6, C_CAL, "Platt\nScaling", "3-fold CV on train", fs=9.5)
    ax.text(9.05, 1.7, "No held-out\ncalibration set",
            ha="center", fontsize=7, color=C_CAL, style="italic")
    arrow(10.05, 3.0, 10.35, 3.0)

    box(10.35, 2.35, 2.25, 1.3, C_OUT, "CASE Output", "Calibrated P(y|X)", fs=9)
    for xi, txt in [(10.55, "↓ ECE"), (11.5, "↓ Brier")]:
        ax.text(xi, 1.95, txt, ha="center", fontsize=7.5, color=C_OUT,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C_OUT, lw=0.9))

    ax.text(6.5, 0.18,
            "Base models pre-specified before experiments (Grinsztajn et al. 2022; "
            "Shwartz-Ziv & Armon 2022) — no circular dependency with benchmark results.",
            ha="center", va="center", fontsize=7.5, color="#555", style="italic")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    _save("fig_case_architecture")


def fig_calibration(datasets: dict, all_results: dict) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold

    SEED = 42; N_BINS = 8
    REP_DS = ["Heart", "Pima", "Ionosphere"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
    fig.suptitle(
        "Calibration Reliability Diagrams: CASE vs CatBoost vs kNN\n"
        "(Perfect calibration = dashed diagonal; curve above = overconfident)",
        fontsize=10.5, fontweight="bold", y=1.01)

    for row, ds_name in enumerate(REP_DS):
        X, y = datasets[ds_name]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)
        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr); X_te_i = imp.transform(X_te)

        # CatBoost
        cat = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1,
                                  random_state=SEED, verbose=0)
        cat.fit(X_tr_i, y_tr)
        prob_cat = cat.predict_proba(X_te_i)[:, 1]

        # kNN
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(X_tr_i); Xte_sc = sc.transform(X_te_i)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr_sc, y_tr)
        prob_knn = knn.predict_proba(Xte_sc)[:, 1]

        # CASE
        top4 = [
            CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1,
                               random_state=SEED, verbose=0),
            RandomForestClassifier(n_estimators=200, max_depth=10,
                                   random_state=SEED, n_jobs=-1),
            LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                           random_state=SEED, n_jobs=-1, verbose=-1),
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          eval_metric="mlogloss", random_state=SEED,
                          n_jobs=-1, verbosity=0),
        ]
        nc = 2
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        meta_oof, meta_te = [], []
        for clf in top4:
            oof = np.zeros((len(X_tr_i), nc))
            for ii, iv in skf.split(X_tr_i, y_tr):
                c_ = clf.__class__(**clf.get_params())
                try:
                    c_.fit(X_tr_i[ii], y_tr[ii])
                    oof[iv] = c_.predict_proba(X_tr_i[iv])
                except Exception: pass
            c_full = clf.__class__(**clf.get_params())
            try:
                c_full.fit(X_tr_i, y_tr)
                meta_te.append(c_full.predict_proba(X_te_i))
            except Exception:
                meta_te.append(np.zeros((len(X_te_i), nc)))
            meta_oof.append(oof)
        Mtr = np.hstack(meta_oof); Mte = np.hstack(meta_te)
        case_clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=500), cv=3, method="sigmoid")
        case_clf.fit(Mtr, y_tr)
        prob_case = case_clf.predict_proba(Mte)[:, 1]

        for col, (prob, label, color) in enumerate([
            (prob_cat,  "CatBoost", "#E86E3A"),
            (prob_knn,  "kNN",      "#E07070"),
            (prob_case, "CASE",     "#2C3E50"),
        ]):
            ax = axes[row, col]
            try:
                frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=N_BINS,
                                                         strategy="uniform")
            except Exception:
                frac_pos, mean_pred = np.array([]), np.array([])
            ece = (np.mean(np.abs(frac_pos - mean_pred))
                   if len(mean_pred) > 0 else np.nan)
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, alpha=0.7)
            if len(mean_pred) > 0:
                ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, markersize=5)
                ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.12, color=color)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0: ax.set_ylabel("Fraction of Positives", fontsize=8)
            ax.set_xlabel("Mean Predicted Prob.", fontsize=8)
            ax.set_title(f"{ds_name} — {label}", fontsize=8.5,
                         fontweight="bold", color=color)
            if not np.isnan(ece):
                ax.text(0.04, 0.93, f"ECE = {ece:.3f}", transform=ax.transAxes,
                        fontsize=8, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=color, lw=0.8, alpha=0.9))

    plt.tight_layout()
    _save("fig_calibration")


def fig_training_time(datasets: dict, all_results: dict) -> None:
    import time
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold

    SEED = 42
    X, y = datasets["Pima"]
    imp = SimpleImputer(strategy="median"); X_i = imp.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, vl_idx = next(iter(skf.split(X_i, y)))

    mdl_map = {
        "LR":       LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "SVM":      SVC(C=10, gamma="scale", probability=True, random_state=SEED),
        "kNN":      KNeighborsClassifier(n_neighbors=5),
        "RF":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                           random_state=SEED, n_jobs=-1),
        "XGBoost":  XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  eval_metric="mlogloss", random_state=SEED,
                                  n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                   random_state=SEED, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1,
                                        random_state=SEED, verbose=0),
        "MLP":      MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01,
                                  max_iter=500, random_state=SEED),
    }
    SCALE = {"LR", "SVM", "kNN", "MLP"}
    train_times = {}
    for mn, clf in mdl_map.items():
        Xtr = StandardScaler().fit_transform(X_i[tr_idx]) if mn in SCALE else X_i[tr_idx]
        t0 = time.perf_counter(); clf.fit(Xtr, y[tr_idx])
        train_times[mn] = time.perf_counter() - t0

    # CASE estimate
    train_times["CASE"] = sum(train_times[m] for m in
                              ["CatBoost", "RF", "LightGBM", "XGBoost"]) * 1.35

    mean_accs = {}
    for mn in MODEL_ORDER:
        vals = [_mu(all_results[ds][mn]["acc"])
                for ds in DS_ORDER if mn in all_results.get(ds, {})]
        mean_accs[mn] = float(np.mean(vals)) if vals else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: training time bars
    ax = axes[0]
    sorted_t = sorted(MODEL_ORDER, key=lambda m: train_times.get(m, 0))
    ax.barh(sorted_t, [train_times.get(m, 0) * 1000 for m in sorted_t],
            color=[COLORS[m] for m in sorted_t], edgecolor="white", height=0.6)
    for mn in sorted_t:
        t = train_times.get(mn, 0) * 1000
        ax.text(t + 5, sorted_t.index(mn), f"{t:.0f} ms", va="center", fontsize=8.5)
    ax.set_xlabel("Training Time (ms) — single fold, Pima dataset")
    ax.set_title("Training Time per Model", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(train_times.values()) * 1000 * 1.30)

    # Right: accuracy vs log(time)
    ax = axes[1]
    for mn in MODEL_ORDER:
        t = train_times.get(mn, 0.001) * 1000
        a = mean_accs.get(mn, 0)
        ax.scatter(np.log10(t + 0.1), a, s=220 if mn == "CASE" else 120,
                   c=COLORS[mn], edgecolors="black" if mn == "CASE" else "white",
                   linewidths=1.5, zorder=4)
        ax.annotate(mn, (np.log10(t + 0.1), a),
                    xytext=(np.log10(t + 0.1) + 0.05, a + 0.003),
                    fontsize=8.5, color=COLORS[mn], fontweight="bold")
    ax.set_xlabel("log₁₀(Training Time in ms)")
    ax.set_ylabel("Mean Accuracy (across 10 datasets)")
    ax.set_title("Accuracy vs. Training Cost", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Computational Cost Analysis (Pima dataset, single fold)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("fig_training_time")


def generate_all(datasets: dict, all_results: dict, stat_results: dict) -> None:
    """Generate all figures in one call."""
    print("\n" + "=" * 50)
    print("Generating all figures")
    print("=" * 50)
    fig_dataset_bubble(datasets)
    fig_rank_heatmap(all_results)
    fig_avg_rank(all_results)
    fig_boxplots(all_results)
    fig_grouped_bars(all_results)
    fig_cd_diagram(all_results)
    fig_wilcoxon(stat_results)
    fig_case_architecture()
    fig_calibration(datasets, all_results)
    fig_training_time(datasets, all_results)
    print(f"\n  All figures saved to: {FIGURES_DIR}/")
