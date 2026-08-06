"""
download_datasets.py
Downloads and saves all 10 benchmark datasets as pickle files.
All datasets are from the UCI Machine Learning Repository via public mirrors.
No institutional access or manual download required.

Usage:
    python data/download_datasets.py
"""

import io
import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder

SAVE_PATH = os.path.join(os.path.dirname(__file__), "datasets.pkl")
JB = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"


def _fetch_csv(url: str, **kwargs) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=20) as r:
        return pd.read_csv(io.BytesIO(r.read()), **kwargs)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_heart() -> tuple:
    """Heart Disease (Cleveland). Binary: 0=no disease, 1=disease."""
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = _fetch_csv(url)
    X = df.drop("target", axis=1).values.astype(float)
    y = df["target"].values.astype(int)
    return X, y


def load_wdbc() -> tuple:
    """Breast Cancer Wisconsin (Diagnostic). Binary: 0=malignant, 1=benign."""
    d = load_breast_cancer()
    return d.data, d.target


def load_haberman() -> tuple:
    """Haberman Survival. Binary: 0=died within 5yr, 1=survived >5yr."""
    df = _fetch_csv(JB + "haberman.csv", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = (df.iloc[:, -1].values == 2).astype(int)
    return X, y


def load_horse() -> tuple:
    """Horse Colic. Binary: 1=survived (no surgery needed), 0=otherwise."""
    df = _fetch_csv(JB + "horse-colic.csv", header=None)
    df = df.replace("?", np.nan)
    y = (df.iloc[:, 23].astype(float) == 1).astype(int).values
    X = (df.drop(df.columns[[23, 24]], axis=1)
           .apply(pd.to_numeric, errors="coerce")
           .values.astype(float))
    return X, y


def load_pima() -> tuple:
    """Pima Indians Diabetes. Binary: 0=no diabetes, 1=diabetes.
    Note: biologically impossible zeros replaced with NaN before imputation."""
    cols = ["preg", "glu", "bp", "skin", "ins", "bmi", "dpf", "age", "label"]
    df = _fetch_csv(JB + "pima-indians-diabetes.csv", names=cols)
    for c in ["glu", "bp", "skin", "ins", "bmi"]:
        df[c] = df[c].replace(0, np.nan)
    return df.drop("label", axis=1).values, df["label"].values.astype(int)


def load_ecoli() -> tuple:
    """E.coli Protein Localisation. 8-class multiclass."""
    df = _fetch_csv(JB + "ecoli.csv", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y


def load_sonar() -> tuple:
    """Sonar. Binary: 0=rock, 1=mine."""
    df = _fetch_csv(JB + "sonar.csv", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = (df.iloc[:, -1] == "M").astype(int).values
    return X, y


def load_german() -> tuple:
    """German Credit. Binary: 0=bad credit, 1=good credit.
    Categorical features are one-hot encoded."""
    df = _fetch_csv(JB + "german.csv", header=None)
    X = pd.get_dummies(df.iloc[:, :-1]).values.astype(float)
    y = (df.iloc[:, -1].values == 2).astype(int)
    return X, y


def load_glass() -> tuple:
    """Glass Identification. 6-class multiclass."""
    df = _fetch_csv(JB + "glass.csv", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y


def load_ionosphere() -> tuple:
    """Ionosphere. Binary: 0=bad signal, 1=good signal."""
    df = _fetch_csv(JB + "ionosphere.csv", header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = (df.iloc[:, -1] == "g").astype(int).values
    return X, y


# ── Dataset registry ──────────────────────────────────────────────────────────

LOADERS = {
    "Heart":      (load_heart,      "Cardiovascular",  "Binary"),
    "WDBC":       (load_wdbc,       "Oncology",        "Binary"),
    "Haberman":   (load_haberman,   "Oncology",        "Binary"),
    "Horse":      (load_horse,      "Surgical",        "Binary"),
    "Pima":       (load_pima,       "Metabolic",       "Binary"),
    "Ecoli":      (load_ecoli,      "Bioinformatics",  "Multi"),
    "Sonar":      (load_sonar,      "Signal Proc.",    "Binary"),
    "German":     (load_german,     "Financial Risk",  "Binary"),
    "Glass":      (load_glass,      "Forensic Sci.",   "Multi"),
    "Ionosphere": (load_ionosphere, "Atmospheric",     "Binary"),
}

DATASET_META = {
    "Heart":      {"domain": "Cardiovascular",  "task": "Binary", "citation": "detrano1989"},
    "WDBC":       {"domain": "Oncology",        "task": "Binary", "citation": "wolberg1995"},
    "Haberman":   {"domain": "Oncology",        "task": "Binary", "citation": "haberman1976"},
    "Horse":      {"domain": "Surgical",        "task": "Binary", "citation": "dua2019"},
    "Pima":       {"domain": "Metabolic",       "task": "Binary", "citation": "smith1988"},
    "Ecoli":      {"domain": "Bioinformatics",  "task": "Multi",  "citation": "nakai1991"},
    "Sonar":      {"domain": "Signal Proc.",    "task": "Binary", "citation": "gorman1988"},
    "German":     {"domain": "Financial Risk",  "task": "Binary", "citation": "hofmann1994"},
    "Glass":      {"domain": "Forensic Sci.",   "task": "Multi",  "citation": "dua2019"},
    "Ionosphere": {"domain": "Atmospheric",     "task": "Binary", "citation": "sigillito1989"},
}


def download_all(save_path: str = SAVE_PATH, verbose: bool = True) -> dict:
    """Download all datasets and save to a pickle file.

    Returns
    -------
    dict mapping dataset name → (X, y) tuples.
    """
    datasets = {}
    print("=" * 60)
    print("Downloading all 10 benchmark datasets")
    print("=" * 60)

    for name, (loader_fn, domain, task) in LOADERS.items():
        try:
            X, y = loader_fn()
            nc = len(np.unique(y))
            try:
                imb = round(np.max(np.bincount(y)) / (len(y) - np.max(np.bincount(y))), 2)
            except Exception:
                imb = "N/A"
            miss = round(np.isnan(X).sum() / X.size * 100, 1)
            datasets[name] = (X, y)
            if verbose:
                print(f"  ✓ {name:<14} {str(X.shape):<14} "
                      f"cls={nc}  imb={imb}  miss={miss}%")
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(datasets, f)

    print(f"\n  Saved {len(datasets)} datasets → {save_path}")
    return datasets


def load_datasets(path: str = SAVE_PATH) -> dict:
    """Load previously downloaded datasets from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at {path}. "
            "Run `python data/download_datasets.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    download_all()
