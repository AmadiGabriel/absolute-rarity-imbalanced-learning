
from __future__ import annotations

from pathlib import Path
import os
import hashlib
import json
import platform
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
import imblearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, precision_score, recall_score, f1_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN

METHODS = ("none", "ros", "smote", "borderline_smote", "adasyn", "class_weight")
MODELS = ("logistic", "random_forest", "svm_rbf")
METRICS = ("auc", "ap", "balanced_accuracy", "mcc", "precision", "recall", "minority_f1")

# Never use Python's built-in hash() for experimental seeds: it is intentionally
# randomized between Python sessions. This function is stable across machines/runs.
def stable_seed(base_seed: int, *parts) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**31 - 1))

def get_classifier(model_name: str, seed: int, class_weight=None):
    if model_name == "logistic":
        return LogisticRegression(max_iter=3000, solver="liblinear",
                                  class_weight=class_weight, random_state=seed)
    if model_name == "random_forest":
        # Respect the CPU allocation supplied by SLURM.  Falling back to 1
        # prevents accidental oversubscription when the script is run outside SLURM.
        n_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      n_jobs=n_jobs, class_weight=class_weight,
                                      random_state=seed)
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight=class_weight, probability=False,
                   random_state=seed)
    raise ValueError(model_name)

def get_sampler(method: str, seed: int, y_train=None):
    if method in ("none", "class_weight"):
        return "passthrough"
    m = int(np.sum(np.asarray(y_train) == 1)) if y_train is not None else 6
    k = max(1, min(5, m - 1))
    m_neighbors = max(1, min(10, m - 1))
    if method == "ros":
        return RandomOverSampler(random_state=seed)
    if method == "smote":
        return SMOTE(random_state=seed, k_neighbors=k)
    if method == "borderline_smote":
        return BorderlineSMOTE(random_state=seed, k_neighbors=k,
                               m_neighbors=m_neighbors)
    if method == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=k)
    raise ValueError(method)

def make_pipeline(model_name: str, method: str, seed: int, y_train=None):
    cw = "balanced" if method == "class_weight" else None
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("sampler", get_sampler(method, seed, y_train=y_train)),
        ("clf", get_classifier(model_name, seed, class_weight=cw)),
    ])

def score_vector(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)

def metric_dict(estimator, X, y):
    y = np.asarray(y).astype(int)
    scores = score_vector(estimator, X)
    pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_eval": int(len(y)),
        "m_eval": int(np.sum(y == 1)),
        "pi_eval": float(np.mean(y == 1)),
        "auc": roc_auc_score(y, scores),
        "ap": average_precision_score(y, scores),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "minority_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def load_local_dataset(path: Path):
    df = pd.read_csv(path)
    if "_target" not in df.columns:
        raise ValueError(f"{path}: expected a '_target' column. Run prepare_data.py first.")
    y = df.pop("_target").astype(int).to_numpy()
    X = df.apply(pd.to_numeric, errors="coerce")
    return X, y

def exact_sample_indices(y, n_minority: int, n_majority: int, rng,
                         allowed_idx=None, exclude_idx=None):
    y = np.asarray(y).astype(int)
    if allowed_idx is None:
        allowed = np.arange(len(y))
    else:
        allowed = np.asarray(allowed_idx, dtype=int)
    if exclude_idx is not None:
        exclude = np.asarray(exclude_idx, dtype=int)
        allowed = np.setdiff1d(allowed, exclude, assume_unique=False)
    idx1 = allowed[y[allowed] == 1]
    idx0 = allowed[y[allowed] == 0]
    if n_minority > len(idx1) or n_majority > len(idx0):
        raise ValueError(
            f"Requested minority={n_minority}, majority={n_majority}; "
            f"available minority={len(idx1)}, majority={len(idx0)}"
        )
    s1 = rng.choice(idx1, n_minority, replace=False)
    s0 = rng.choice(idx0, n_majority, replace=False)
    idx = np.concatenate([s1, s0])
    rng.shuffle(idx)
    return idx

def take_rows(X, y, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx]
    return X[idx], np.asarray(y)[idx]

def exact_sample(X, y, n_minority: int, n_majority: int, rng,
                 allowed_idx=None, exclude_idx=None):
    idx = exact_sample_indices(y, n_minority, n_majority, rng,
                               allowed_idx=allowed_idx, exclude_idx=exclude_idx)
    Xs, ys = take_rows(X, y, idx)
    return Xs, ys

def write_manifest(out_path: Path, config: dict, extra: dict | None = None):
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "config": config,
    }
    if extra:
        payload["extra"] = extra
    out_path.write_text(json.dumps(payload, indent=2, default=str))
