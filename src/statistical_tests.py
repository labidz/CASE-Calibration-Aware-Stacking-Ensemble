"""
statistical_tests.py
Non-parametric statistical tests for multi-dataset model comparison.

Tests implemented
-----------------
1. Friedman test      — global null: all models perform equivalently
2. Wilcoxon signed-rank — pairwise: each baseline vs CASE, Bonferroni corrected
3. Nemenyi critical difference — CD for post-hoc ranking visualisation

Protocol (from Demšar 2006)
----------------------------
- Run Friedman test first.
- Conduct post-hoc pairwise tests ONLY if Friedman is significant (p < 0.05).
- Apply Bonferroni correction over the number of pairwise comparisons.
"""

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ALPHA = 0.05  # significance threshold


def compute_per_dataset_means(
    all_results: dict,
    dataset_order: list,
    model_order: list,
    metric: str = "acc",
) -> dict:
    """Return per-dataset mean score for each model.

    Parameters
    ----------
    all_results   : {dataset: {model: {metric: [fold_values]}}}
    dataset_order : ordered list of dataset names
    model_order   : ordered list of model names
    metric        : metric key (acc, f1, auc, mcc)

    Returns
    -------
    dict mapping model_name → list of per-dataset means (same order as dataset_order)
    """
    per_model = {}
    for mn in model_order:
        per_model[mn] = []
        for ds in dataset_order:
            if mn in all_results.get(ds, {}):
                vals = [v for v in all_results[ds][mn][metric] if v is not None and not np.isnan(v)]
                per_model[mn].append(float(np.mean(vals)) if vals else np.nan)
            else:
                per_model[mn].append(np.nan)
    return per_model


def friedman_test(per_model: dict, model_order: list) -> tuple:
    """Run the Friedman test across all models.

    Returns
    -------
    (chi2_stat, p_value)
    """
    groups = [np.array(per_model[mn]) for mn in model_order]
    # Remove datasets where any model has NaN
    mask = np.all(~np.isnan(np.column_stack(groups)), axis=1)
    groups_clean = [g[mask] for g in groups]
    if sum(mask) < 3:
        return np.nan, np.nan
    try:
        stat, pval = friedmanchisquare(*groups_clean)
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def wilcoxon_vs_case(
    per_model: dict,
    model_order: list,
    reference: str = "CASE",
    bonferroni_n: int = None,
) -> dict:
    """Wilcoxon signed-rank test comparing each model against the reference.

    Only call this AFTER confirming the Friedman test is significant.

    Parameters
    ----------
    per_model    : output of compute_per_dataset_means()
    model_order  : ordered list of model names
    reference    : model to compare against (default 'CASE')
    bonferroni_n : number of comparisons for Bonferroni correction.
                   Defaults to len(model_order) - 1.

    Returns
    -------
    dict mapping model_name → dict with 'raw_p', 'adj_p', 'significant'
    """
    if bonferroni_n is None:
        bonferroni_n = len(model_order) - 1

    ref_vals = np.array(per_model[reference])
    results = {}

    for mn in model_order:
        if mn == reference:
            continue
        a = np.array(per_model[mn])
        mask = ~(np.isnan(a) | np.isnan(ref_vals))
        if mask.sum() < 5:
            results[mn] = {"raw_p": np.nan, "adj_p": np.nan, "significant": False}
            continue
        try:
            _, raw_p = wilcoxon(a[mask], ref_vals[mask])
            adj_p = min(raw_p * bonferroni_n, 1.0)
            results[mn] = {
                "raw_p":      float(raw_p),
                "adj_p":      float(adj_p),
                "significant": adj_p < ALPHA,
            }
        except Exception:
            results[mn] = {"raw_p": np.nan, "adj_p": np.nan, "significant": False}

    return results


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """Compute the Nemenyi critical difference.

    CD = q_alpha * sqrt(k*(k+1) / (6*N))

    Parameters
    ----------
    k     : number of models/classifiers
    N     : number of datasets
    alpha : significance level (only 0.05 implemented)

    Returns
    -------
    Critical difference value
    """
    # Studentised range critical values from Demšar (2006) Table 5, alpha=0.05
    q_values = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    if alpha != 0.05:
        raise ValueError("Only alpha=0.05 is supported.")
    if k not in q_values:
        raise ValueError(f"k must be in {list(q_values.keys())}; got {k}.")
    q = q_values[k]
    return q * np.sqrt(k * (k + 1) / (6 * N))


def compute_average_ranks(
    all_results: dict,
    dataset_order: list,
    model_order: list,
    metric: str = "acc",
) -> dict:
    """Compute average competition-style rank for each model across all datasets.

    Competition-style: tied values share the average of the tied ranks.

    Returns
    -------
    dict mapping model_name → average rank (float)
    """
    rank_store = {mn: [] for mn in model_order}

    for ds in dataset_order:
        scores = {}
        for mn in model_order:
            if mn in all_results.get(ds, {}):
                vals = [v for v in all_results[ds][mn][metric]
                        if v is not None and not np.isnan(v)]
                if vals:
                    scores[mn] = float(np.mean(vals))

        if not scores:
            continue

        # Sort descending (higher = better)
        sorted_models = sorted(scores, key=lambda x: -scores[x])
        i = 0
        while i < len(sorted_models):
            j = i
            while (j < len(sorted_models) - 1 and
                   abs(scores[sorted_models[j + 1]] - scores[sorted_models[i]]) < 1e-9):
                j += 1
            avg_rank = (i + 1 + j + 1) / 2
            for mn in sorted_models[i: j + 1]:
                rank_store[mn].append(avg_rank)
            i = j + 1

    return {mn: float(np.mean(rank_store[mn])) if rank_store[mn] else np.nan
            for mn in model_order}


def run_all_tests(
    all_results: dict,
    dataset_order: list,
    model_order: list,
) -> dict:
    """Run the complete statistical testing pipeline for accuracy and macro-F1.

    Returns a dict with full test results for both metrics.

    Following the protocol of Demšar (2006):
    - Friedman test is run for both metrics.
    - Post-hoc Wilcoxon comparisons are run ONLY for metrics where
      Friedman is significant (p < 0.05).
    """
    output = {}

    for metric in ["acc", "f1"]:
        per_model = compute_per_dataset_means(all_results, dataset_order, model_order, metric)
        chi2, pval = friedman_test(per_model, model_order)

        result = {
            "metric":       metric,
            "friedman_chi2": chi2,
            "friedman_p":    pval,
            "significant":   pval < ALPHA if not np.isnan(pval) else False,
            "wilcoxon":      None,
        }

        if result["significant"]:
            result["wilcoxon"] = wilcoxon_vs_case(per_model, model_order)
        else:
            result["wilcoxon"] = None  # not reported per protocol

        output[metric] = result
        sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
        print(f"  {metric.upper():<5} Friedman χ²={chi2:.3f}  p={pval:.4f}  [{sig_str}]")
        if result["wilcoxon"]:
            for mn, r in result["wilcoxon"].items():
                star = "**" if r["adj_p"] < 0.01 else ("*" if r["adj_p"] < 0.05 else "ns")
                print(f"         vs {mn:<10} adj_p={r['adj_p']:.4f}  {star}")

    return output
