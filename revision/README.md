# CASE benchmark — revised experiments (SNCS-D-26-06872, revision 1)

Code for "Benchmarking Classifiers on Small and Medium-Sized Tabular Data: A Statistical
and Ablation Study of a Calibration-Aware Stacking Ensemble".

## Files
| File | Purpose |
|---|---|
| `data.py` | Loaders for the 10 core (UCI) and 16 extended (PMLB / OpenML-CC18) datasets, incl. all corrections (E.coli rare classes, Horse Colic leakage columns, missing-value codes) |
| `pipeline.py` | Models, per-model preprocessing (one-hot for LR/SVM/kNN/MLP, integer codes for trees), Protocol A/B imbalance handling, stacking and calibration utilities |
| `run_bench.py` | Repeated stratified CV for all baselines, CASE and all ablation variants (shared base-model fits per fold) |
| `run_hpo.py` | Nested-CV random-search hyper-parameter optimisation |
| `run_automl.py` | AutoGluon-Tabular and TabPFN (v1) on identical folds (separate Python 3.12 env) |
| `metrics.py`, `stats.py` | Metrics (incl. top-label ECE) and statistics (Friedman, Iman–Davenport, Nemenyi CD, Wilcoxon + Holm/Bonferroni, rank-biserial, bootstrap CIs) |
| `analyze.py` | Generates every table, figure and number of the paper from the stored per-fold predictions |
| `fig_arch.py` | Architecture figure |

## Data
Core datasets: `heart_kb22.csv` (github.com/kb22/Heart-Disease-Prediction), `haberman.csv`, `horse-colic.csv`,
`pima-indians-diabetes.csv`, `ecoli.csv`, `sonar.csv`, `german.csv`, `glass.csv`, `ionosphere.csv`
(github.com/jbrownlee/Datasets), WDBC from scikit-learn. Extended datasets: PMLB
`datasets/<name>/<name>.tsv.gz` and `metadata.yaml`. Put them in `$CASE_RAW` (default `data_raw/`, PMLB files in `data_raw/pmlb/`).

## Reproduce
```bash
CORE="Heart WDBC Haberman Horse Pima Ecoli Sonar German Glass Ionosphere"
EXT="contraceptive_method car_evaluation yeast wine_quality_red mfeat_fourier segmentation splice kr_vs_kp hypothyroid spambase churn phoneme page_blocks satimage magic adult"
python run_bench.py --datasets $CORE --protocol A --repeats 5 --tag core --save-oof
python run_bench.py --datasets $CORE --protocol B --repeats 5 --tag core
python run_bench.py --datasets $EXT  --protocol A --repeats 2 --tag ext
python run_hpo.py   --datasets $CORE
python run_automl.py --method TabPFN    --datasets $CORE     # Python 3.12 env with torch
python run_automl.py --method AutoGluon --time-limit 30 --datasets $CORE
python analyze.py && python fig_arch.py
```
Seeds: 42 (models, inner splits), 42–46 (outer repeats), 142 (OOF split of tuned CASE), 0 (subsampling).
Results (per-fold class-probability matrices, fold indices, timings) are written as pickles to `$CASE_OUT`
(default `results/`); the ones used in the paper are in `results.zip`.

TabPFN 0.1.11 with recent libraries: in `tabpfn/layer.py` import `Optional` from `typing`, `Tensor` from `torch`
and `Module, MultiheadAttention, Linear, Dropout, LayerNorm` from `torch.nn`; in
`tabpfn/scripts/transformer_prediction_interface.py` replace `force_all_finite` by `ensure_all_finite`.
