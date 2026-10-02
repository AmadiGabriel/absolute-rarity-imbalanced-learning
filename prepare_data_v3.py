
"""
Robust one-time dataset preparation for the Beyond Imbalance Ratio study.

Changes in this version
-----------------------
1. Existing prepared CSV files are skipped.
2. APS Failure is downloaded directly from the official UCI archive ZIP.
3. SECOM is downloaded directly from the official UCI archive rather than
   relying on ucimlrepo, because the current package can return an empty
   targets table for this dataset.
4. Missing SECOM values remain NaN and are imputed later INSIDE each
   training fold by the modelling pipeline.

Usage:
    python3 prepare_data_v3.py
"""

from pathlib import Path
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

APS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00414/to_uci.zip"

SECOM_ZIP_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SECOM_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def already_done(name):
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, nrows=5)
            if "_target" in df.columns:
                print(f"{name:15s} -> already present; skipping")
                return True
        except Exception:
            pass
    return False


def save_binary(X, y, name):
    X = pd.DataFrame(X).reset_index(drop=True).copy()
    ydf = pd.DataFrame(y).reset_index(drop=True)

    if ydf.shape[1] != 1:
        raise ValueError(f"{name}: expected one target column, got {ydf.shape[1]}")

    y = ydf.iloc[:, 0]

    if len(X) != len(y):
        raise ValueError(f"{name}: X/y row mismatch: {len(X)} vs {len(y)}")

    counts = y.value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError(f"{name}: expected binary target, got {counts.to_dict()}")

    minority_label = counts.idxmin()
    y_bin = (y == minority_label).astype(int)

    X.columns = [str(c) for c in X.columns]
    X["_target"] = y_bin.to_numpy()

    out = DATA_DIR / f"{name}.csv"
    X.to_csv(out, index=False)

    print(
        f"{name:15s} -> {out.name}: "
        f"n={len(X):,}, p={X.shape[1]-1}, "
        f"minority={int(X['_target'].sum()):,} "
        f"({X['_target'].mean():.4%})"
    )


def fetch_openml_dataset(data_id, name):
    if already_done(name):
        return

    from sklearn.datasets import fetch_openml

    print(f"Downloading OpenML dataset {data_id}: {name}")
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    save_binary(bunch.data, bunch.target, name)


def fetch_uci_dataset(data_id, name):
    if already_done(name):
        return

    from ucimlrepo import fetch_ucirepo

    print(f"Downloading UCI dataset {data_id}: {name}")
    ds = fetch_ucirepo(id=data_id)
    save_binary(ds.data.features, ds.data.targets, name)


def fetch_aps_direct():
    name = "aps_failure"
    if already_done(name):
        return

    print("Downloading APS Failure directly from official UCI archive")

    with urllib.request.urlopen(APS_URL, timeout=180) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = zf.namelist()
        train_candidates = [
            m for m in members
            if m.endswith("aps_failure_training_set.csv")
        ]
        if not train_candidates:
            raise FileNotFoundError(
                "Could not find aps_failure_training_set.csv in UCI ZIP."
            )

        with zf.open(train_candidates[0]) as fh:
            df = pd.read_csv(
                fh,
                skiprows=20,
                na_values=["na", "NA", ""],
                low_memory=False
            )

    df.columns = [str(c).strip() for c in df.columns]

    if "class" not in df.columns:
        raise ValueError("APS file parsed but no 'class' column was found.")

    y_raw = df.pop("class").astype(str).str.strip().str.lower()
    valid = y_raw.isin(["pos", "neg"])
    if not valid.all():
        bad = y_raw[~valid].value_counts(dropna=False).to_dict()
        raise ValueError(f"Unexpected APS class labels: {bad}")

    y = (y_raw == "pos").astype(int)
    X = df.apply(pd.to_numeric, errors="coerce")

    X["_target"] = y.to_numpy()
    out = DATA_DIR / f"{name}.csv"
    X.to_csv(out, index=False)

    print(
        f"{name:15s} -> {out.name}: "
        f"n={len(X):,}, p={X.shape[1]-1}, "
        f"minority={int(X['_target'].sum()):,} "
        f"({X['_target'].mean():.4%})"
    )


def _download_bytes(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def fetch_secom_direct():
    name = "secom"
    if already_done(name):
        return

    print("Downloading SECOM directly from official UCI archive")

    data_bytes = None
    label_bytes = None

    # Try the modern UCI ZIP first.
    try:
        zip_bytes = _download_bytes(SECOM_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = zf.namelist()

            data_candidates = [
                m for m in members if Path(m).name.lower() == "secom.data"
            ]
            label_candidates = [
                m for m in members if Path(m).name.lower() == "secom_labels.data"
            ]

            if data_candidates and label_candidates:
                data_bytes = zf.read(data_candidates[0])
                label_bytes = zf.read(label_candidates[0])

    except Exception as e:
        print(f"  modern SECOM ZIP unavailable ({type(e).__name__}); using legacy files")

    # Fall back to the legacy official UCI raw files.
    if data_bytes is None or label_bytes is None:
        data_bytes = _download_bytes(SECOM_DATA_URL)
        label_bytes = _download_bytes(SECOM_LABEL_URL)

    # Feature matrix: whitespace-separated numerical values with NaN tokens.
    X = pd.read_csv(
        io.BytesIO(data_bytes),
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "NA"],
        engine="python"
    )

    # Labels file format:
    # -1 "19/07/2008 11:55:00"
    #  1 "19/07/2008 13:17:00"
    # We intentionally use only the class label; timestamp is metadata,
    # not a predictor in this study.
    labels_text = label_bytes.decode("utf-8", errors="replace").splitlines()
    labels = []
    for line in labels_text:
        line = line.strip()
        if not line:
            continue
        labels.append(int(line.split()[0]))

    y = pd.Series(labels, name="_target_raw")

    if len(X) != len(y):
        raise ValueError(
            f"SECOM X/y row mismatch: X={len(X)}, y={len(y)}"
        )

    if set(y.unique()) != {-1, 1}:
        raise ValueError(f"Unexpected SECOM labels: {sorted(y.unique())}")

    # Official documentation defines 1=fail and -1=pass.
    y_bin = (y == 1).astype(int)

    X.columns = [f"feature_{i:03d}" for i in range(X.shape[1])]
    X["_target"] = y_bin.to_numpy()

    out = DATA_DIR / "secom.csv"
    X.to_csv(out, index=False)

    print(
        f"{name:15s} -> {out.name}: "
        f"n={len(X):,}, p={X.shape[1]-1}, "
        f"minority={int(X['_target'].sum()):,} "
        f"({X['_target'].mean():.4%})"
    )


def main():
    fetch_openml_dataset(1597, "creditcard")
    fetch_openml_dataset(310, "mammography")
    fetch_openml_dataset(311, "oil_spill")

    fetch_uci_dataset(372, "htru2")

    fetch_aps_direct()
    fetch_secom_direct()

    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()
