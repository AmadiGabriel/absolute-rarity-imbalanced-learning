
"""
Controlled synthetic study.

Training-rarity scenarios:
  A1 fixed total N, varying m_train
  A2 fixed minority proportion, varying m_train  [isolates absolute rarity]
  A3 fixed m_train, varying minority proportion  [isolates relative rarity]

Validation-rarity scenarios:
  V1 fixed validation N, varying m_val
  V2 fixed validation proportion, varying m_val [isolates absolute rarity]
  V3 fixed m_val, varying validation proportion [isolates relative rarity]

A large, untouched oracle test set is reserved first. For every fitted model,
training and validation observations are disjoint from the oracle and from each
other. Different experimental cells may reuse population observations; each
Monte-Carlo repetition is generated from a fresh population.

Usage:
    python run_synthetic.py --profile pilot
    python run_synthetic.py --profile full
    python run_synthetic.py --profile pilot --repeats 1 --models logistic --methods none smote
"""
from pathlib import Path
import argparse, json, shutil
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from study_common import (
    METHODS, MODELS, make_pipeline, metric_dict, score_vector, exact_sample_indices,
    take_rows, stable_seed, write_manifest
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--repeats", type=int, default=None,
                   help="Override configured repeats; useful for smoke tests.")
    p.add_argument("--models", nargs="+", choices=MODELS,
                   default=["logistic", "random_forest"])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--resume", action="store_true",
                   help="Resume from deterministic per-cell checkpoints.")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing checkpoints for this profile and start again.")
    p.add_argument("--finalize-only", action="store_true",
                   help="Do not fit models; only consolidate existing checkpoint fragments.")
    return p.parse_args()

def population(seed, p, class_sep, flip_y, n, n_informative=5, n_redundant=2):
    """
    Keep the amount of true signal fixed while p changes.

    This makes p=10, 50, 100 a genuine nuisance-dimensionality manipulation:
    added dimensions are noise rather than extra informative variables.
    """
    n_informative = int(min(max(2, n_informative), p))
    n_redundant = int(min(max(0, n_redundant), p - n_informative))
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=2,
        n_clusters_per_class=2, weights=[0.5, 0.5],
        class_sep=class_sep, flip_y=flip_y,
        random_state=seed, shuffle=True,
    )
    return X.astype(np.float32), y.astype(int)


def ap_components(y, scores):
    """
    Prepare tie-aware cumulative counts once so AP can be standardized
    to any target prevalence without re-sorting the oracle scores.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ys = y[order]
    ss = scores[order]
    distinct = np.where(np.diff(ss))[0]
    idx = np.r_[distinct, len(ss) - 1]
    tps = np.cumsum(ys)[idx].astype(float)
    fps = (idx + 1 - tps).astype(float)
    dtp = np.diff(np.r_[0.0, tps])
    npos = float(ys.sum())
    nneg = float(len(ys) - ys.sum())
    return tps, fps, dtp, npos, nneg


def prevalence_standardized_ap(components, target_pi):
    """
    Average precision for the same score ranking standardized to target_pi.

    This is equivalent to assigning constant class weights so the weighted
    prevalence equals target_pi. It lets validation AP be compared with an
    oracle AP at the SAME prevalence.
    """
    tps, fps, dtp, npos, nneg = components
    target_pi = float(target_pi)
    if not (0.0 < target_pi < 1.0):
        return np.nan
    wpos = target_pi / npos
    wneg = (1.0 - target_pi) / nneg
    denom = wpos * tps + wneg * fps
    precision = np.divide(
        wpos * tps, denom,
        out=np.zeros_like(tps, dtype=float),
        where=denom > 0
    )
    return float(np.sum((dtp / npos) * precision))

def train_conditions(cfg):
    for m in cfg["minority_counts"]:
        yield "A1_fixed_total", m, cfg["fixed_total_n"] - m, m / cfg["fixed_total_n"]
    pi = cfg["fixed_ratio"]
    for m in cfg["minority_counts"]:
        maj = int(round(m * (1 - pi) / pi))
        yield "A2_fixed_ratio", m, maj, m / (m + maj)
    m = cfg["fixed_minority_for_ratio_scenario"]
    for pi in cfg["ratios"]:
        maj = int(round(m * (1 - pi) / pi))
        yield "A3_fixed_minority", m, maj, m / (m + maj)

def validation_conditions(cfg):
    # V1: total validation N fixed; absolute and relative rarity change together.
    N = cfg["validation_fixed_total_n"]
    for m in cfg["validation_minority_counts"]:
        if m < N:
            yield "V1_fixed_total", m, N - m, m / N
    # V2: prevalence fixed; pure absolute-rarity manipulation.
    pi = cfg["validation_ratio"]
    for m in cfg["validation_minority_counts"]:
        maj = int(round(m * (1 - pi) / pi))
        yield "V2_fixed_ratio", m, maj, m / (m + maj)
    # V3: m_val fixed; pure relative-rarity manipulation.
    m = cfg["validation_fixed_minority"]
    for pi in cfg["validation_ratios"]:
        maj = int(round(m * (1 - pi) / pi))
        yield "V3_fixed_minority", m, maj, m / (m + maj)


def _safe_token(x):
    """Filename-safe deterministic token."""
    if isinstance(x, float):
        s = f"{x:.8g}"
    else:
        s = str(x)
    return (s.replace("-", "m")
              .replace(".", "p")
              .replace("+", "")
              .replace(" ", "_"))


def _cell_id(rep, p, sep, flip, scenario, m_train, nmaj_train):
    return (
        f"rep{rep:03d}_p{p:03d}"
        f"_sep{_safe_token(sep)}_flip{_safe_token(flip)}"
        f"_{scenario}_m{m_train:06d}_maj{nmaj_train:07d}"
    )


def _atomic_write_csv(rows, path):
    """
    Write a complete cell fragment atomically.
    Empty cells are represented by a zero-byte file and handled at consolidation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tmp, index=False)
    else:
        tmp.write_text("")
    tmp.replace(path)


def _atomic_write_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _read_csv_fragments(folder):
    frames = []
    for path in sorted(folder.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _consolidate(checkpoint_root, profile):
    """Rebuild the canonical result CSVs entirely from deterministic fragments."""
    oracle = _read_csv_fragments(checkpoint_root / "oracle")
    val = _read_csv_fragments(checkpoint_root / "validation")
    failures = _read_csv_fragments(checkpoint_root / "failures")

    oracle.to_csv(RESULTS / f"synthetic_oracle_{profile}.csv", index=False)
    val.to_csv(RESULTS / f"synthetic_validation_{profile}.csv", index=False)
    failures.to_csv(RESULTS / f"synthetic_failures_{profile}.csv", index=False)

    return len(oracle), len(val), len(failures)


def main():
    args = parse_args()
    cfg_all = json.loads((ROOT / "config.json").read_text())
    cfg = cfg_all["synthetic"]
    base_seed = cfg_all["random_seed"]

    reps = args.repeats or (
        cfg["pilot_repeats"] if args.profile == "pilot"
        else cfg["full_repeats"]
    )
    dims = (
        cfg["dimensions_pilot"] if args.profile == "pilot"
        else cfg["dimensions_full"]
    )
    seps = (
        cfg["class_separations_pilot"] if args.profile == "pilot"
        else cfg["class_separations_full"]
    )
    flips = (
        cfg["flip_y_pilot"] if args.profile == "pilot"
        else cfg["flip_y_full"]
    )

    checkpoint_root = RESULTS / "checkpoints" / f"synthetic_{args.profile}"
    done_dir = checkpoint_root / "done"

    run_signature = {
        "profile": args.profile,
        "repeats": reps,
        "models": list(args.models),
        "methods": list(args.methods),
        "dimensions": list(dims),
        "class_separations": list(seps),
        "flip_y": list(flips),
        "synthetic_config": cfg,
        "random_seed": base_seed,
        "script_version": "synthetic_final_checkpoint_v1",
        "ap_validation_reference": "prevalence-matched oracle AP",
        "n_informative": cfg.get("n_informative", 5),
        "n_redundant": cfg.get("n_redundant", 2),
    }
    signature_path = checkpoint_root / "checkpoint_manifest.json"

    if args.overwrite and checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)

    if args.finalize_only:
        if not checkpoint_root.exists():
            raise SystemExit(
                f"No checkpoint directory found at {checkpoint_root}"
            )
        no, nv, nf = _consolidate(checkpoint_root, args.profile)
        print(
            f"Finalized checkpoint fragments: oracle={no:,}, "
            f"validation={nv:,}, failures={nf:,}"
        )
        return

    if checkpoint_root.exists() and signature_path.exists():
        existing = json.loads(signature_path.read_text())
        if existing != run_signature:
            raise SystemExit(
                "Existing checkpoints were created with different settings.\n"
                "Use the same command with --resume, or use --overwrite to "
                "deliberately start a new run."
            )
        if not args.resume and any(done_dir.glob("*.done")):
            raise SystemExit(
                f"Existing checkpoints found in {checkpoint_root}.\n"
                "Use --resume to continue, or --overwrite to start again."
            )
    else:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            json.dumps(run_signature, indent=2, sort_keys=True),
            signature_path
        )

    for sub in ("oracle", "validation", "failures", "done"):
        (checkpoint_root / sub).mkdir(parents=True, exist_ok=True)

    total_cells = (
        reps * len(dims) * len(seps) * len(flips)
        * sum(1 for _ in train_conditions(cfg))
    )
    existing_done = len(list(done_dir.glob("*.done")))
    print(
        f"Profile={args.profile}; total train cells={total_cells:,}; "
        f"already complete={existing_done:,}; remaining={total_cells-existing_done:,}"
    )

    completed_this_run = 0

    for rep in range(reps):
        for p in dims:
            for sep in seps:
                for flip in flips:
                    # If all cells for this population are complete, skip generation.
                    population_cells = [
                        (scenario, m_train, nmaj_train, pi_train,
                         _cell_id(rep, p, sep, flip, scenario, m_train, nmaj_train))
                        for scenario, m_train, nmaj_train, pi_train
                        in train_conditions(cfg)
                    ]
                    incomplete = [
                        x for x in population_cells
                        if not (done_dir / f"{x[4]}.done").exists()
                    ]
                    if not incomplete:
                        print(
                            f"skip complete population rep={rep+1}/{reps}, "
                            f"p={p}, sep={sep}, flip={flip}"
                        )
                        continue

                    pop_seed = stable_seed(
                        base_seed, "population", rep, p, sep, flip
                    )
                    Xpop, ypop = population(
                        pop_seed, p, sep, flip, cfg["population_n"],
                        n_informative=cfg.get("n_informative", 5),
                        n_redundant=cfg.get("n_redundant", 2)
                    )

                    all_idx = np.arange(len(ypop))
                    rng_oracle = np.random.default_rng(
                        stable_seed(
                            base_seed, "oracle", rep, p, sep, flip
                        )
                    )
                    oracle_idx = exact_sample_indices(
                        ypop,
                        cfg["oracle_positive_n"],
                        cfg["oracle_negative_n"],
                        rng_oracle
                    )
                    Xoracle, yoracle = take_rows(
                        Xpop, ypop, oracle_idx
                    )
                    dev_idx = np.setdiff1d(
                        all_idx, oracle_idx, assume_unique=False
                    )

                    for (
                        scenario, m_train, nmaj_train, pi_train, cell_id
                    ) in population_cells:

                        done_path = done_dir / f"{cell_id}.done"
                        if done_path.exists():
                            continue

                        cell_oracle_rows = []
                        cell_val_rows = []
                        cell_fail_rows = []

                        cell_seed = stable_seed(
                            base_seed, "traincell", rep, p, sep, flip,
                            scenario, m_train, nmaj_train
                        )
                        rng_train = np.random.default_rng(cell_seed)

                        try:
                            train_idx = exact_sample_indices(
                                ypop, m_train, nmaj_train, rng_train,
                                allowed_idx=dev_idx
                            )
                        except Exception as e:
                            cell_fail_rows.append({
                                "stage": "train_sampling",
                                "rep": rep,
                                "scenario": scenario,
                                "p": p,
                                "class_sep": sep,
                                "flip_y": flip,
                                "m_train": m_train,
                                "nmaj_train": nmaj_train,
                                "error": repr(e),
                            })
                            _atomic_write_csv(
                                cell_oracle_rows,
                                checkpoint_root / "oracle" / f"{cell_id}.csv"
                            )
                            _atomic_write_csv(
                                cell_val_rows,
                                checkpoint_root / "validation" / f"{cell_id}.csv"
                            )
                            _atomic_write_csv(
                                cell_fail_rows,
                                checkpoint_root / "failures" / f"{cell_id}.csv"
                            )
                            _atomic_write_text("done\n", done_path)
                            completed_this_run += 1
                            continue

                        Xtr, ytr = take_rows(Xpop, ypop, train_idx)
                        val_pool = np.setdiff1d(
                            dev_idx, train_idx, assume_unique=False
                        )

                        fitted = {}
                        oracle_ref = {}

                        for model in args.models:
                            for method in args.methods:
                                seed = stable_seed(
                                    base_seed, "fit", rep, p, sep, flip,
                                    scenario, m_train, model, method
                                )
                                try:
                                    est = make_pipeline(
                                        model, method, seed, y_train=ytr
                                    )
                                    est.fit(Xtr, ytr)
                                    fitted[(model, method)] = est

                                    met = metric_dict(
                                        est, Xoracle, yoracle
                                    )
                                    oscores = score_vector(
                                        est, Xoracle
                                    )
                                    oracle_ref[(model, method)] = {
                                        "auc": float(met["auc"]),
                                        "ap_components": ap_components(
                                            yoracle, oscores
                                        ),
                                    }

                                    cell_oracle_rows.append({
                                        "rep": rep,
                                        "scenario": scenario,
                                        "p": p,
                                        "n_informative": cfg.get(
                                            "n_informative", 5
                                        ),
                                        "n_redundant": cfg.get(
                                            "n_redundant", 2
                                        ),
                                        "class_sep": sep,
                                        "flip_y": flip,
                                        "m_train": m_train,
                                        "nmaj_train": nmaj_train,
                                        "n_train": m_train + nmaj_train,
                                        "pi_train": pi_train,
                                        "model": model,
                                        "method": method,
                                        **met,
                                    })
                                except Exception as e:
                                    cell_fail_rows.append({
                                        "stage": "fit_or_oracle",
                                        "rep": rep,
                                        "scenario": scenario,
                                        "p": p,
                                        "class_sep": sep,
                                        "flip_y": flip,
                                        "m_train": m_train,
                                        "nmaj_train": nmaj_train,
                                        "model": model,
                                        "method": method,
                                        "error": repr(e),
                                    })

                        if scenario == "A2_fixed_ratio":
                            for (
                                vscenario, m_val, nmaj_val, pi_val
                            ) in validation_conditions(cfg):

                                v_seed = stable_seed(
                                    base_seed, "val", rep, p, sep, flip,
                                    m_train, vscenario, m_val, nmaj_val
                                )
                                rng_val = np.random.default_rng(v_seed)

                                try:
                                    val_idx = exact_sample_indices(
                                        ypop, m_val, nmaj_val, rng_val,
                                        allowed_idx=val_pool
                                    )
                                    Xv, yv = take_rows(
                                        Xpop, ypop, val_idx
                                    )
                                except Exception as e:
                                    cell_fail_rows.append({
                                        "stage": "validation_sampling",
                                        "rep": rep,
                                        "validation_scenario": vscenario,
                                        "p": p,
                                        "class_sep": sep,
                                        "flip_y": flip,
                                        "m_train": m_train,
                                        "m_val": m_val,
                                        "nmaj_val": nmaj_val,
                                        "error": repr(e),
                                    })
                                    continue

                                for model in args.models:
                                    for method in args.methods:
                                        est = fitted.get((model, method))
                                        if est is None:
                                            continue
                                        try:
                                            metv = metric_dict(
                                                est, Xv, yv
                                            )
                                            oref = oracle_ref[
                                                (model, method)
                                            ]
                                            oracle_ap_matched = (
                                                prevalence_standardized_ap(
                                                    oref["ap_components"],
                                                    pi_val
                                                )
                                            )
                                            cell_val_rows.append({
                                                "rep": rep,
                                                "training_scenario": scenario,
                                                "validation_scenario": vscenario,
                                                "p": p,
                                                "class_sep": sep,
                                                "flip_y": flip,
                                                "m_train": m_train,
                                                "pi_train": pi_train,
                                                "m_val": m_val,
                                                "nmaj_val": nmaj_val,
                                                "n_val": m_val + nmaj_val,
                                                "pi_val": pi_val,
                                                "model": model,
                                                "method": method,
                                                "oracle_auc": oref["auc"],
                                                "oracle_ap_matched": (
                                                    oracle_ap_matched
                                                ),
                                                **metv,
                                            })
                                        except Exception as e:
                                            cell_fail_rows.append({
                                                "stage": "validation_score",
                                                "rep": rep,
                                                "validation_scenario": (
                                                    vscenario
                                                ),
                                                "p": p,
                                                "class_sep": sep,
                                                "flip_y": flip,
                                                "m_train": m_train,
                                                "m_val": m_val,
                                                "model": model,
                                                "method": method,
                                                "error": repr(e),
                                            })

                        # Write all fragments first. The .done marker is written LAST.
                        # If the process stops before .done is created, --resume will
                        # recompute and safely overwrite this deterministic cell.
                        _atomic_write_csv(
                            cell_oracle_rows,
                            checkpoint_root / "oracle" / f"{cell_id}.csv"
                        )
                        _atomic_write_csv(
                            cell_val_rows,
                            checkpoint_root / "validation" / f"{cell_id}.csv"
                        )
                        _atomic_write_csv(
                            cell_fail_rows,
                            checkpoint_root / "failures" / f"{cell_id}.csv"
                        )
                        _atomic_write_text("done\n", done_path)

                        completed_this_run += 1
                        total_done = existing_done + completed_this_run
                        print(
                            f"[{total_done:,}/{total_cells:,}] completed "
                            f"rep={rep+1}/{reps}, p={p}, sep={sep}, "
                            f"flip={flip}, {scenario}, "
                            f"m_train={m_train}"
                        )

    # Rebuild canonical files from checkpoints.
    no, nv, nf = _consolidate(checkpoint_root, args.profile)

    write_manifest(
        RESULTS / f"synthetic_manifest_{args.profile}.json",
        cfg_all,
        {
            "models": args.models,
            "methods": args.methods,
            "repeats": reps,
            "n_informative": cfg.get("n_informative", 5),
            "n_redundant": cfg.get("n_redundant", 2),
            "ap_validation_reference": "prevalence-matched oracle AP",
            "checkpointed": True,
            "checkpoint_granularity": "training-condition cell",
        }
    )

    print(
        "\nSynthetic results consolidated successfully.\n"
        f"Oracle rows: {no:,}\n"
        f"Validation rows: {nv:,}\n"
        f"Failure rows: {nf:,}\n"
        f"Checkpoints: {checkpoint_root}\n"
        f"Next: python3 analyze_results_v2.py --profile {args.profile}"
    )


if __name__ == "__main__":
    main()
