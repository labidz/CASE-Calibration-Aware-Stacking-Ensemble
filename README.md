# Tabular ML Benchmark with CASE

**A Comprehensive Benchmark of Machine Learning Models for Tabular Data:
Multi-Domain Evaluation and a Calibration-Aware Stacking Ensemble**

---

## Overview

This repository contains the complete experimental code for the paper. It benchmarks
**8 established ML models** across **10 public datasets** spanning 7 domains, and
proposes **CASE** (Calibration-Aware Stacking Ensemble), a three-layer ensemble that
stacks gradient-boosted tree models and applies Platt scaling for calibrated output.

### Key Results

| Model     | Avg. Accuracy Rank ↓ |
|-----------|----------------------|
| **CASE**  | **2.70**             |
| CatBoost  | 3.60                 |
| RF        | 3.90                 |
| LightGBM  | 5.30                 |
| SVM       | 4.80                 |
| LR        | 5.50                 |
| XGBoost   | 5.70                 |
| MLP       | 5.70                 |
| kNN       | 7.80                 |

- Friedman test: χ² = 23.61, **p = 0.003** (significant)
- CASE significantly outperforms 6/8 baselines (Wilcoxon, Bonferroni corrected)

---

## Repository Structure

```
benchmark-case/
├── README.md
├── requirements.txt
├── reproduce.sh              # One-command full reproduction
├── run_benchmark.py          # Main entry point
├── data/
│   └── download_datasets.py  # Downloads all 10 UCI datasets
├── src/
│   ├── preprocessing.py      # Preprocessing pipelines (leak-free, inside CV)
│   ├── models.py             # All 8 baseline model configurations
│   ├── case.py               # CASE model implementation
│   ├── evaluation.py         # CV protocol + metric computation
│   ├── statistical_tests.py  # Friedman + Wilcoxon + Nemenyi CD
│   └── figures.py            # All publication figure generation
├── results/                  # Auto-created; stores raw JSON + CSV tables
└── figures/                  # Auto-created; stores all figures (PDF + PNG)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/benchmark-case.git
cd benchmark-case
pip install -r requirements.txt
```

### 2. One-command reproduction

```bash
bash reproduce.sh
```

This downloads all datasets, runs all experiments, performs statistical tests,
and generates all figures. Total runtime: ~30–60 minutes (CPU).

### 3. Step-by-step

```bash
# Step 1: Download all datasets
python data/download_datasets.py

# Step 2: Run benchmark (all models × all datasets)
python run_benchmark.py

# Step 3: Generate figures only (requires existing results)
python run_benchmark.py --figures-only
```

---

## Datasets

All datasets are downloaded automatically from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu) via public mirrors.
No manual download is required.

| Dataset             | Domain          | N    | p  | Classes | Imbalance |
|---------------------|-----------------|------|----|---------|-----------|
| Heart (Cleveland)   | Cardiovascular  | 303  | 13 | 2       | 1.20      |
| WDBC                | Oncology        | 569  | 30 | 2       | 1.68      |
| Haberman            | Oncology        | 306  |  3 | 2       | 2.78      |
| Horse Colic         | Surgical        | 300  | 26 | 2       | 1.75      |
| Pima Diabetes       | Metabolic       | 768  |  8 | 2       | 1.87      |
| E.coli Localisation | Bioinformatics  | 336  |  7 | 8       | 7.33      |
| Sonar               | Signal Proc.    | 208  | 60 | 2       | 1.14      |
| German Credit       | Financial Risk  | 1000 | 61 | 2       | 2.33      |
| Glass               | Forensic Sci.   | 214  |  9 | 6       | 5.50      |
| Ionosphere          | Atmospheric     | 351  | 34 | 2       | 1.79      |

---

## Models

| Family          | Models                              |
|-----------------|-------------------------------------|
| Classical ML    | Logistic Regression, SVM, kNN       |
| Tree Ensemble   | Random Forest, XGBoost, LightGBM, CatBoost |
| Neural          | MLP                                 |
| **Proposed**    | **CASE** (Calibration-Aware Stacking Ensemble) |

All baseline hyperparameters are fixed to best-practice defaults (see Appendix of paper).
No per-dataset tuning was performed to ensure fair comparison.

---

## Evaluation Protocol

- **5-fold stratified cross-validation × 2 repeats** = 10 evaluation runs per model per dataset
- All preprocessing inside folds (no data leakage)
- Metrics: Accuracy, Macro-F1, AUC-ROC, MCC
- Statistical tests: Friedman test + Wilcoxon signed-rank (Bonferroni corrected)

---

## CASE Architecture

```
Input Tabular Data
       │
       ▼
┌──────────────────────────────────────┐
│  LAYER 1: Base Models (pre-specified)│
│  CatBoost | RF | LightGBM | XGBoost  │
│  → Out-of-fold probabilities (3-fold)│
└──────────────────────────────────────┘
       │  Stack [4 × K] probabilities
       ▼
┌──────────────────────────────────────┐
│  LAYER 2: Meta-Learner               │
│  Logistic Regression on OOF probs    │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  LAYER 3: Platt Calibration          │
│  Fitted via 3-fold CV on train fold  │
└──────────────────────────────────────┘
       │
       ▼
Calibrated class probabilities
```

Base models were **pre-specified** before any experiments based on prior literature
(Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022), not selected post-hoc from
benchmark results.

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{amin2025benchmark,
  title   = {A Comprehensive Benchmark of Machine Learning Models for Tabular Data:
             Multi-Domain Evaluation and a Calibration-Aware Stacking Ensemble},
  author  = {Amin, Md. Rohul and Aktar, Sumya and Rana, Md. Mehedi Rahman and Akter, Zakia},
  journal = {[Journal Name]},
  year    = {2025}
}
```

---

## License

MIT License. See `LICENSE` for details.
