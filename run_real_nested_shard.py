"""
One outer-fold shard of the real-data repeated nested-CV experiment.

Scientifically identical to run_real_nested.py:
  * same RepeatedStratifiedKFold outer splits
  * same StratifiedKFold inner splits
  * same deterministic seeds
  * same hyperparameter grids
  * same preprocessing/resampling pipeline
  * same metrics

One invocation evaluates one outer_index only.
"""

from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)

from study_common import (
    METHODS,
    MODELS,
    make_pipeline,
    metric_dict,
    load_local_dataset,
    stable_seed,
    write_manifest,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "real_nested_shards"
RESULTS.mkdir(parents=True, exist_ok=True)

DATASETS = {
    k: DATA / f"{k}.csv"
    for k in [
        "creditcard",
        "mammography",
        "htru2",
        "aps_failure",
        "secom",
        "oil_spill",
    ]
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="full",
    )

    p.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    p.add_argument(
        "--outer-index",
        required=True,
        type=int,
    )

    p.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=["logistic", "random_forest"],
    )

    p.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )

    return p.parse_args()


def param_grid(model):
    if model == "logistic":
        return {
            "clf__C": [0.1, 1.0, 10.0]
        }

    if model == "random_forest":
        return {
            "clf__max_features": ["sqrt", 0.5],
            "clf__min_samples_leaf": [1, 3, 8],
        }

    if model == "svm_rbf":
        return {
            "clf__C": [0.5, 1.0, 5.0],
            "clf__gamma": ["scale", 0.01],
        }

    raise ValueError(model)


def allocated_cpus():
    try:
        return max(
            1,
            int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        )
    except ValueError:
        return 1


def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    args = parse_args()

    cfg_all = json.loads(
        (ROOT / "config.json").read_text()
    )

    cfg = cfg_all["real"]
    base = cfg_all["random_seed"]
    cpus = allocated_cpus()

    k_req = (
        cfg["pilot_outer_splits"]
        if args.profile == "pilot"
        else cfg["full_outer_splits"]
    )

    r_req = (
        cfg["pilot_outer_repeats"]
        if args.profile == "pilot"
        else cfg["full_outer_repeats"]
    )

    X, y = load_local_dataset(
        DATASETS[args.dataset]
    )

    nmin = int(y.sum())

    k = max(
        2,
        min(k_req, nmin)
    )

    n_outer = k * r_req

    if args.outer_index < 0 or args.outer_index >= n_outer:
        raise ValueError(
            f"outer-index must be 0..{n_outer - 1}"
        )

    outer = RepeatedStratifiedKFold(
        n_splits=k,
        n_repeats=r_req,
        random_state=stable_seed(
            base,
            "outer",
            args.dataset,
        ),
    )

    selected = None

    for oi, (tr, te) in enumerate(
        outer.split(X, y)
    ):
        if oi == args.outer_index:
            selected = (tr, te)
            break

    if selected is None:
        raise RuntimeError(
            "Could not locate requested outer fold"
        )

    tr, te = selected

    repeat = args.outer_index // k
    fold = args.outer_index % k

    Xtr = X.iloc[tr]
    Xte = X.iloc[te]

    ytr = y[tr]
    yte = y[te]

    mtr = int(ytr.sum())
    mte = int(yte.sum())

    k_inner = max(
        2,
        min(
            cfg["inner_splits"],
            mtr,
        )
    )

    inner = StratifiedKFold(
        n_splits=k_inner,
        shuffle=True,
        random_state=stable_seed(
            base,
            "inner",
            args.dataset,
            repeat,
            fold,
        ),
    )

    out_path = (
        RESULTS /
        f"real_nested_{args.profile}_"
        f"{args.dataset}_outer{args.outer_index:02d}.csv"
    )

    fail_path = (
        RESULTS /
        f"real_nested_failures_{args.profile}_"
        f"{args.dataset}_outer{args.outer_index:02d}.csv"
    )

    manifest_path = (
        RESULTS /
        f"real_nested_manifest_{args.profile}_"
        f"{args.dataset}_outer{args.outer_index:02d}.json"
    )

    if out_path.exists() and out_path.stat().st_size > 0:
        previous = pd.read_csv(out_path)
    else:
        previous = pd.DataFrame()

    done = set()

    if len(previous):
        done = set(
            zip(
                previous["model"],
                previous["method"],
            )
        )

    rows = []
    failures = []

    print("=" * 70, flush=True)
    print("Nested-CV outer-fold shard", flush=True)
    print("Dataset:", args.dataset, flush=True)
    print(
        f"outer_index={args.outer_index}/{n_outer - 1}",
        flush=True,
    )
    print(
        f"repeat={repeat}, fold={fold}",
        flush=True,
    )
    print(
        f"train n={len(ytr):,}, minority={mtr}",
        flush=True,
    )
    print(
        f"test n={len(yte):,}, minority={mte}",
        flush=True,
    )
    print("Grid-search workers:", cpus, flush=True)
    print("=" * 70, flush=True)

    for model in args.models:

        for method in args.methods:

            if (model, method) in done:
                print(
                    f"SKIP completed: {model} / {method}",
                    flush=True,
                )
                continue

            seed = stable_seed(
                base,
                "realfit",
                args.dataset,
                repeat,
                fold,
                model,
                method,
            )

            print(
                f"START {model} / {method}",
                flush=True,
            )

            try:
                pipe = make_pipeline(
                    model,
                    method,
                    seed,
                    y_train=ytr,
                )

                # GridSearchCV owns parallelism.
                if model == "random_forest":
                    pipe.set_params(
                        clf__n_jobs=1
                    )

                search = GridSearchCV(
                    pipe,
                    param_grid(model),
                    scoring="roc_auc",
                    cv=inner,
                    n_jobs=cpus,
                    refit=True,
                    error_score=np.nan,
                    pre_dispatch=cpus,
                    return_train_score=False,
                )

                search.fit(
                    Xtr,
                    ytr,
                )

                met = metric_dict(
                    search.best_estimator_,
                    Xte,
                    yte,
                )

                rows.append({
                    "dataset": args.dataset,

                    "outer_repeat": repeat,
                    "outer_fold": fold,
                    "outer_index":
                        args.outer_index,

                    "outer_k": k,
                    "outer_repeats": r_req,

                    "n": len(y),
                    "p": X.shape[1],

                    "minority_total": nmin,
                    "prevalence_total":
                        nmin / len(y),

                    "n_train": len(ytr),
                    "m_train": mtr,
                    "pi_train":
                        mtr / len(ytr),

                    "n_test": len(yte),
                    "m_test": mte,
                    "pi_test":
                        mte / len(yte),

                    "model": model,
                    "method": method,

                    "best_inner_auc":
                        search.best_score_,

                    "best_params":
                        repr(search.best_params_),

                    **met,
                })

                current = pd.concat(
                    [
                        previous,
                        pd.DataFrame(rows),
                    ],
                    ignore_index=True,
                )

                current = (
                    current
                    .drop_duplicates(
                        ["model", "method"],
                        keep="last",
                    )
                    .sort_values(
                        ["model", "method"]
                    )
                )

                atomic_csv(
                    current,
                    out_path,
                )

                done.add(
                    (model, method)
                )

                print(
                    f"DONE {model} / {method}; "
                    f"{len(done)}/"
                    f"{len(args.models)*len(args.methods)}",
                    flush=True,
                )

            except Exception as e:

                failures.append({
                    "dataset":
                        args.dataset,
                    "outer_repeat":
                        repeat,
                    "outer_fold":
                        fold,
                    "outer_index":
                        args.outer_index,
                    "m_train":
                        mtr,
                    "m_test":
                        mte,
                    "model":
                        model,
                    "method":
                        method,
                    "error":
                        repr(e),
                })

                atomic_csv(
                    pd.DataFrame(failures),
                    fail_path,
                )

                print(
                    f"FAIL {model} / {method}: {repr(e)}",
                    flush=True,
                )

    final = (
        pd.read_csv(out_path)
        if out_path.exists()
        else pd.DataFrame()
    )

    write_manifest(
        manifest_path,
        cfg_all,
        {
            "dataset": args.dataset,
            "outer_index":
                args.outer_index,
            "outer_repeat":
                repeat,
            "outer_fold":
                fold,
            "models":
                args.models,
            "methods":
                args.methods,
            "gridsearch_workers":
                cpus,
            "random_forest_workers_per_fit":
                1,
            "expected_rows":
                len(args.models) *
                len(args.methods),
            "completed_rows":
                len(final),
        },
    )

    print(
        f"SHARD COMPLETE: {len(final)}/12 rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
