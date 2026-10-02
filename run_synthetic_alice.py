
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
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from study_common_alice import (
    METHODS, MODELS, make_pipeline, metric_dict, score_vector, exact_sample_indices,
    take_rows, stable_seed, write_manifest
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["pilot", "full"], default="full")
    p.add_argument("--rep", type=int, default=None,
                   help="1-based repetition number. Defaults to SLURM_ARRAY_TASK_ID.")
    p.add_argument("--models", nargs="+", choices=MODELS,
                   default=["logistic", "random_forest"])
    p.add_argument("--methods", nargs="+", choices=METHODS,
                   default=list(METHODS))
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


def _scientific_signature(cfg_all, profile, models, methods):
    cfg = cfg_all["synthetic"]
    reps = cfg["pilot_repeats"] if profile == "pilot" else cfg["full_repeats"]
    dims = cfg["dimensions_pilot"] if profile == "pilot" else cfg["dimensions_full"]
    seps = cfg["class_separations_pilot"] if profile == "pilot" else cfg["class_separations_full"]
    flips = cfg["flip_y_pilot"] if profile == "pilot" else cfg["flip_y_full"]
    # Intentionally identical to the Mac checkpoint signature, so copied
    # checkpoints are directly reusable on ALICE.
    return {
        "profile": profile,
        "repeats": reps,
        "models": list(models),
        "methods": list(methods),
        "dimensions": list(dims),
        "class_separations": list(seps),
        "flip_y": list(flips),
        "synthetic_config": cfg,
        "random_seed": cfg_all["random_seed"],
        "script_version": "synthetic_final_checkpoint_v1",
        "ap_validation_reference": "prevalence-matched oracle AP",
        "n_informative": cfg.get("n_informative", 5),
        "n_redundant": cfg.get("n_redundant", 2),
    }


def main():
    args = parse_args()
    cfg_all = json.loads((ROOT / "config.json").read_text())
    cfg = cfg_all["synthetic"]
    base_seed = cfg_all["random_seed"]

    reps = cfg["pilot_repeats"] if args.profile == "pilot" else cfg["full_repeats"]
    dims = cfg["dimensions_pilot"] if args.profile == "pilot" else cfg["dimensions_full"]
    seps = cfg["class_separations_pilot"] if args.profile == "pilot" else cfg["class_separations_full"]
    flips = cfg["flip_y_pilot"] if args.profile == "pilot" else cfg["flip_y_full"]

    rep_one_based = args.rep
    if rep_one_based is None:
        env_rep = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_rep is None:
            raise SystemExit("Provide --rep N, or run under a SLURM array with SLURM_ARRAY_TASK_ID set.")
        rep_one_based = int(env_rep)
    if not (1 <= rep_one_based <= reps):
        raise SystemExit(f"Repetition must be between 1 and {reps}; got {rep_one_based}.")
    rep = rep_one_based - 1

    checkpoint_root = RESULTS / "checkpoints" / f"synthetic_{args.profile}"
    done_dir = checkpoint_root / "done"
    signature_path = checkpoint_root / "checkpoint_manifest.json"
    expected = _scientific_signature(cfg_all, args.profile, args.models, args.methods)

    if not signature_path.exists():
        raise SystemExit(
            f"Missing {signature_path}. Run prepare_synthetic_alice.py before submitting the array."
        )
    existing = json.loads(signature_path.read_text())
    if existing != expected:
        raise SystemExit(
            "Checkpoint settings do not match this ALICE run. Do not overwrite them. "
            "Check config.json/models/methods, or start in a clean project copy."
        )

    for sub in ("oracle", "validation", "failures", "done"):
        (checkpoint_root / sub).mkdir(parents=True, exist_ok=True)

    cells = list(train_conditions(cfg))
    per_rep_total = len(dims) * len(seps) * len(flips) * len(cells)
    local_done_before = 0
    for p in dims:
        for sep in seps:
            for flip in flips:
                for scenario, m_train, nmaj_train, _ in cells:
                    cid = _cell_id(rep, p, sep, flip, scenario, m_train, nmaj_train)
                    local_done_before += int((done_dir / f"{cid}.done").exists())

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(
        f"ALICE task repetition={rep_one_based}/{reps}; cpus={cpus}; "
        f"cells complete={local_done_before}/{per_rep_total}; "
        f"remaining={per_rep_total-local_done_before}", flush=True
    )

    completed_now = 0
    for p in dims:
        for sep in seps:
            for flip in flips:
                population_cells = [
                    (scenario, m_train, nmaj_train, pi_train,
                     _cell_id(rep, p, sep, flip, scenario, m_train, nmaj_train))
                    for scenario, m_train, nmaj_train, pi_train in cells
                ]
                incomplete = [x for x in population_cells
                              if not (done_dir / f"{x[4]}.done").exists()]
                if not incomplete:
                    continue

                pop_seed = stable_seed(base_seed, "population", rep, p, sep, flip)
                Xpop, ypop = population(
                    pop_seed, p, sep, flip, cfg["population_n"],
                    n_informative=cfg.get("n_informative", 5),
                    n_redundant=cfg.get("n_redundant", 2)
                )

                all_idx = np.arange(len(ypop))
                rng_oracle = np.random.default_rng(
                    stable_seed(base_seed, "oracle", rep, p, sep, flip)
                )
                oracle_idx = exact_sample_indices(
                    ypop, cfg["oracle_positive_n"], cfg["oracle_negative_n"], rng_oracle
                )
                Xoracle, yoracle = take_rows(Xpop, ypop, oracle_idx)
                dev_idx = np.setdiff1d(all_idx, oracle_idx, assume_unique=False)

                for scenario, m_train, nmaj_train, pi_train, cell_id in population_cells:
                    done_path = done_dir / f"{cell_id}.done"
                    if done_path.exists():
                        continue

                    cell_oracle_rows, cell_val_rows, cell_fail_rows = [], [], []
                    cell_seed = stable_seed(
                        base_seed, "traincell", rep, p, sep, flip,
                        scenario, m_train, nmaj_train
                    )
                    rng_train = np.random.default_rng(cell_seed)

                    try:
                        train_idx = exact_sample_indices(
                            ypop, m_train, nmaj_train, rng_train, allowed_idx=dev_idx
                        )
                    except Exception as e:
                        cell_fail_rows.append({
                            "stage":"train_sampling", "rep":rep, "scenario":scenario,
                            "p":p, "class_sep":sep, "flip_y":flip,
                            "m_train":m_train, "nmaj_train":nmaj_train, "error":repr(e)
                        })
                        _atomic_write_csv(cell_oracle_rows, checkpoint_root/"oracle"/f"{cell_id}.csv")
                        _atomic_write_csv(cell_val_rows, checkpoint_root/"validation"/f"{cell_id}.csv")
                        _atomic_write_csv(cell_fail_rows, checkpoint_root/"failures"/f"{cell_id}.csv")
                        _atomic_write_text("done\n", done_path)
                        completed_now += 1
                        continue

                    Xtr, ytr = take_rows(Xpop, ypop, train_idx)
                    val_pool = np.setdiff1d(dev_idx, train_idx, assume_unique=False)
                    fitted, oracle_ref = {}, {}

                    for model in args.models:
                        for method in args.methods:
                            seed = stable_seed(
                                base_seed, "fit", rep, p, sep, flip,
                                scenario, m_train, model, method
                            )
                            try:
                                est = make_pipeline(model, method, seed, y_train=ytr)
                                est.fit(Xtr, ytr)
                                fitted[(model, method)] = est
                                met = metric_dict(est, Xoracle, yoracle)
                                oscores = score_vector(est, Xoracle)
                                oracle_ref[(model, method)] = {
                                    "auc": float(met["auc"]),
                                    "ap_components": ap_components(yoracle, oscores),
                                }
                                cell_oracle_rows.append({
                                    "rep":rep, "scenario":scenario, "p":p,
                                    "n_informative":cfg.get("n_informative",5),
                                    "n_redundant":cfg.get("n_redundant",2),
                                    "class_sep":sep, "flip_y":flip,
                                    "m_train":m_train, "nmaj_train":nmaj_train,
                                    "n_train":m_train+nmaj_train, "pi_train":pi_train,
                                    "model":model, "method":method, **met
                                })
                            except Exception as e:
                                cell_fail_rows.append({
                                    "stage":"fit_or_oracle", "rep":rep, "scenario":scenario,
                                    "p":p, "class_sep":sep, "flip_y":flip,
                                    "m_train":m_train, "nmaj_train":nmaj_train,
                                    "model":model, "method":method, "error":repr(e)
                                })

                    if scenario == "A2_fixed_ratio":
                        for vscenario, m_val, nmaj_val, pi_val in validation_conditions(cfg):
                            v_seed = stable_seed(
                                base_seed, "val", rep, p, sep, flip,
                                m_train, vscenario, m_val, nmaj_val
                            )
                            rng_val = np.random.default_rng(v_seed)
                            try:
                                val_idx = exact_sample_indices(
                                    ypop, m_val, nmaj_val, rng_val, allowed_idx=val_pool
                                )
                                Xv, yv = take_rows(Xpop, ypop, val_idx)
                            except Exception as e:
                                cell_fail_rows.append({
                                    "stage":"validation_sampling", "rep":rep,
                                    "validation_scenario":vscenario, "p":p,
                                    "class_sep":sep, "flip_y":flip, "m_train":m_train,
                                    "m_val":m_val, "nmaj_val":nmaj_val, "error":repr(e)
                                })
                                continue

                            for model in args.models:
                                for method in args.methods:
                                    est = fitted.get((model, method))
                                    if est is None:
                                        continue
                                    try:
                                        metv = metric_dict(est, Xv, yv)
                                        oref = oracle_ref[(model, method)]
                                        oracle_ap_matched = prevalence_standardized_ap(
                                            oref["ap_components"], pi_val
                                        )
                                        cell_val_rows.append({
                                            "rep":rep, "training_scenario":scenario,
                                            "validation_scenario":vscenario, "p":p,
                                            "class_sep":sep, "flip_y":flip,
                                            "m_train":m_train, "pi_train":pi_train,
                                            "m_val":m_val, "nmaj_val":nmaj_val,
                                            "n_val":m_val+nmaj_val, "pi_val":pi_val,
                                            "model":model, "method":method,
                                            "oracle_auc":oref["auc"],
                                            "oracle_ap_matched":oracle_ap_matched, **metv
                                        })
                                    except Exception as e:
                                        cell_fail_rows.append({
                                            "stage":"validation_score", "rep":rep,
                                            "validation_scenario":vscenario, "p":p,
                                            "class_sep":sep, "flip_y":flip,
                                            "m_train":m_train, "m_val":m_val,
                                            "model":model, "method":method, "error":repr(e)
                                        })

                    _atomic_write_csv(cell_oracle_rows, checkpoint_root/"oracle"/f"{cell_id}.csv")
                    _atomic_write_csv(cell_val_rows, checkpoint_root/"validation"/f"{cell_id}.csv")
                    _atomic_write_csv(cell_fail_rows, checkpoint_root/"failures"/f"{cell_id}.csv")
                    _atomic_write_text("done\n", done_path)
                    completed_now += 1
                    current = local_done_before + completed_now
                    print(
                        f"rep {rep_one_based}: [{current}/{per_rep_total}] "
                        f"p={p}, sep={sep}, flip={flip}, {scenario}, m_train={m_train}",
                        flush=True
                    )

    print(
        f"Repetition {rep_one_based} complete. "
        f"New cells={completed_now}; total cells for repetition={per_rep_total}.",
        flush=True
    )


if __name__ == "__main__":
    main()
