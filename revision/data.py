"""
data.py -- dataset loaders for the revised benchmark.

Core suite (10 UCI datasets): loaded from the same public mirrors used in the
original submission; several corrections were applied in the revision:
  * E.coli: the two classes with only two instances each (imL, imS) are removed
    (they cannot be stratified across 5 folds); 332 instances, 6 classes remain.
  * Horse Colic: target = surgical lesion (attr. 24). The hospital number (ID),
    the outcome (post-hoc), the three lesion-type codes (derived from the target)
    and cp_data (no information) are removed.  Nominal attributes are declared
    categorical.
  * German Credit: raw 20 attributes are kept (13 categorical); one-hot
    encoding is done inside the CV folds, only for scale-sensitive models.
  * Heart: 'ca' == 4 and 'thal' == 0 are the missing-value codes of this mirror
    and are set to NaN (6 + 1 = 7 missing entries, as in the original UCI file).
  * Pima: implausible zeros in five features are set to NaN.

Extended suite (16 datasets, n >= 1,473): PMLB copies of OpenML/UCI datasets.
Datasets with more than MAX_N instances are stratified-subsampled to MAX_N.
"""
import os
import numpy as np
import pandas as pd
import yaml

RAW = os.environ.get("CASE_RAW", "data_raw")
MAX_N = 5000
SUBSAMPLE_SEED = 0

# ---------------------------------------------------------------------------
# metadata: domain, UCI donation year, source citation key
CORE_META = {
    "Heart":      dict(domain="Medicine",          year=1988, task="Binary"),
    "WDBC":       dict(domain="Medicine",          year=1995, task="Binary"),
    "Haberman":   dict(domain="Medicine",          year=1999, task="Binary"),
    "Horse":      dict(domain="Veterinary med.",   year=1989, task="Binary"),
    "Pima":       dict(domain="Medicine",          year=1990, task="Binary"),
    "Ecoli":      dict(domain="Molecular biology", year=1996, task="Multi"),
    "Sonar":      dict(domain="Physics/signals",   year=1988, task="Binary"),
    "German":     dict(domain="Finance",           year=1994, task="Binary"),
    "Glass":      dict(domain="Forensics/materials", year=1987, task="Multi"),
    "Ionosphere": dict(domain="Physics/signals",   year=1988, task="Binary"),
}
CORE_ORDER = list(CORE_META)

EXT_META = {
    "contraceptive_method": dict(short="CMC",        domain="Social science",     year=1997),
    "car_evaluation":       dict(short="Car",        domain="Decision support",   year=1997),
    "yeast":                dict(short="Yeast",      domain="Molecular biology",  year=1996),
    "wine_quality_red":     dict(short="Wine-red",   domain="Food chemistry",     year=2009),
    "mfeat_fourier":        dict(short="Mfeat-fou",  domain="Pattern recognition", year=1998),
    "segmentation":         dict(short="Segment",    domain="Computer vision",    year=1990),
    "splice":               dict(short="Splice",     domain="Molecular biology",  year=1992),
    "kr_vs_kp":             dict(short="KR-vs-KP",   domain="Games",              year=1989),
    "hypothyroid":          dict(short="Hypothyroid", domain="Medicine",          year=1987),
    "spambase":             dict(short="Spambase",   domain="Text/e-mail",        year=1999),
    "churn":                dict(short="Churn",      domain="Business",           year=2003),
    "phoneme":              dict(short="Phoneme",    domain="Speech",             year=1993),
    "page_blocks":          dict(short="Page-blocks", domain="Document analysis", year=1995),
    "satimage":             dict(short="Satimage",   domain="Remote sensing",     year=1993),
    "magic":                dict(short="MAGIC",      domain="Physics/signals",    year=2007),
    "adult":                dict(short="Adult",      domain="Socio-economic",     year=1996),
}
EXT_ORDER = list(EXT_META)


def _jb(name, **kw):
    return pd.read_csv(os.path.join(RAW, name), **kw)


def load_heart():
    df = pd.read_csv(os.path.join(RAW, "heart_kb22.csv"), encoding="utf-8-sig")
    df.loc[df["ca"] == 4, "ca"] = np.nan
    df.loc[df["thal"] == 0, "thal"] = np.nan
    y = df.pop("target").values.astype(int)
    cats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    return df.astype(float), y, cats


def load_wdbc():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    return d.data.astype(float), d.target.values.astype(int), []


def load_haberman():
    df = _jb("haberman.csv", header=None, names=["age", "year", "nodes", "status"])
    y = (df.pop("status").values == 2).astype(int)
    return df.astype(float), y, []


HORSE_COLS = ["surgery", "age", "hospital_number", "rectal_temp", "pulse",
              "respiratory_rate", "temp_extremities", "peripheral_pulse",
              "mucous_membrane", "capillary_refill", "pain", "peristalsis",
              "abdominal_distention", "nasogastric_tube", "nasogastric_reflux",
              "nasogastric_reflux_ph", "rectal_exam_feces", "abdomen",
              "packed_cell_volume", "total_protein", "abdomo_appearance",
              "abdomo_protein", "outcome", "surgical_lesion", "lesion_1",
              "lesion_2", "lesion_3", "cp_data"]


def load_horse():
    df = _jb("horse-colic.csv", header=None, names=HORSE_COLS, na_values="?")
    y = (df["surgical_lesion"].values == 1).astype(int)
    df = df.drop(columns=["hospital_number", "outcome", "surgical_lesion",
                          "lesion_1", "lesion_2", "lesion_3", "cp_data"])
    cats = ["surgery", "age", "temp_extremities", "peripheral_pulse",
            "mucous_membrane", "capillary_refill", "pain", "peristalsis",
            "abdominal_distention", "nasogastric_tube", "nasogastric_reflux",
            "rectal_exam_feces", "abdomen", "abdomo_appearance"]
    return df.astype(float), y, cats


def load_pima():
    cols = ["preg", "glu", "bp", "skin", "ins", "bmi", "dpf", "age", "label"]
    df = _jb("pima-indians-diabetes.csv", names=cols)
    for c in ["glu", "bp", "skin", "ins", "bmi"]:
        df[c] = df[c].replace(0, np.nan)
    y = df.pop("label").values.astype(int)
    return df.astype(float), y, []


def load_ecoli():
    df = _jb("ecoli.csv", header=None)
    df = df[~df[7].isin(["imL", "imS"])].reset_index(drop=True)
    classes = sorted(df[7].unique())
    y = df[7].map({c: i for i, c in enumerate(classes)}).values.astype(int)
    X = df.iloc[:, :7].astype(float)
    X.columns = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
    return X, y, []


def load_sonar():
    df = _jb("sonar.csv", header=None)
    y = (df.iloc[:, -1] == "M").astype(int).values
    X = df.iloc[:, :-1].astype(float)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return X, y, []


def load_german():
    df = _jb("german.csv", header=None)
    y = (df.iloc[:, -1].values == 2).astype(int)
    X = df.iloc[:, :-1].copy()
    X.columns = [f"A{i+1}" for i in range(20)]
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats:  # string codes -> integer codes (category identity only)
        X[c] = X[c].astype("category").cat.codes.astype(float)
    return X.astype(float), y, cats


def load_glass():
    df = _jb("glass.csv", header=None)
    classes = sorted(df.iloc[:, -1].unique())
    y = df.iloc[:, -1].map({c: i for i, c in enumerate(classes)}).values.astype(int)
    X = df.iloc[:, :-1].astype(float)
    X.columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    return X, y, []


def load_ionosphere():
    df = _jb("ionosphere.csv", header=None)
    y = (df.iloc[:, -1] == "g").astype(int).values
    X = df.iloc[:, :-1].astype(float)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return X, y, []


CORE_LOADERS = {
    "Heart": load_heart, "WDBC": load_wdbc, "Haberman": load_haberman,
    "Horse": load_horse, "Pima": load_pima, "Ecoli": load_ecoli,
    "Sonar": load_sonar, "German": load_german, "Glass": load_glass,
    "Ionosphere": load_ionosphere,
}


def load_pmlb(name):
    df = pd.read_csv(os.path.join(RAW, "pmlb", f"{name}.tsv.gz"), sep="\t")
    meta = yaml.safe_load(open(os.path.join(RAW, "pmlb", f"{name}.yaml")))
    y_raw = df.pop("target").values
    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw).astype(int)
    types = {f["name"]: f["type"] for f in meta["features"]}
    cats = [c for c in df.columns if types.get(c) in ("categorical", "binary")]
    X = df.astype(float)
    n_orig = len(y)
    if len(y) > MAX_N:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(y)), train_size=MAX_N,
                                  stratify=y, random_state=SUBSAMPLE_SEED)
        idx = np.sort(idx)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y, cats, n_orig


def load(name):
    """Return X (DataFrame), y (int array), list of categorical columns, info."""
    if name in CORE_LOADERS:
        X, y, cats = CORE_LOADERS[name]()
        info = dict(CORE_META[name], n_orig=len(y), suite="core")
    else:
        X, y, cats, n_orig = load_pmlb(name)
        info = dict(EXT_META[name], n_orig=n_orig, suite="extended")
        info["task"] = "Binary" if len(np.unique(y)) == 2 else "Multi"
    return X, y, cats, info


def describe(name):
    X, y, cats, info = load(name)
    counts = np.bincount(y)
    nominal = [c for c in cats if X[c].nunique() > 2]
    return dict(
        dataset=name, n=len(y), n_orig=info["n_orig"], p=X.shape[1],
        n_cat=len(cats), n_nominal=len(nominal), classes=len(counts),
        imbalance=counts.max() / counts.min(), min_class=int(counts.min()),
        majority_rate=counts.max() / counts.sum(),
        missing_pct=100 * X.isna().values.mean(),
        rows_with_missing_pct=100 * X.isna().any(axis=1).mean(),
        **{k: info[k] for k in ("domain", "year", "task")},
    )


if __name__ == "__main__":
    rows = [describe(n) for n in CORE_ORDER + EXT_ORDER]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string())
