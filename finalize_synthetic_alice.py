from pathlib import Path
import argparse, json
import pandas as pd
from study_common_alice import write_manifest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_fragments(folder):
    frames=[]
    for path in sorted(folder.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--profile",choices=["pilot","full"],default="full")
    args=p.parse_args()

    cfg_all=json.loads((ROOT/"config.json").read_text())
    cfg=cfg_all["synthetic"]
    reps=cfg["pilot_repeats"] if args.profile=="pilot" else cfg["full_repeats"]
    dims=cfg["dimensions_pilot"] if args.profile=="pilot" else cfg["dimensions_full"]
    seps=cfg["class_separations_pilot"] if args.profile=="pilot" else cfg["class_separations_full"]
    flips=cfg["flip_y_pilot"] if args.profile=="pilot" else cfg["flip_y_full"]
    cells_per_population = len(cfg["minority_counts"])*2 + len(cfg["ratios"])
    expected_done = reps*len(dims)*len(seps)*len(flips)*cells_per_population

    root=RESULTS/"checkpoints"/f"synthetic_{args.profile}"
    actual_done=len(list((root/"done").glob("*.done")))
    if actual_done != expected_done:
        raise SystemExit(
            f"Not all cells are complete: {actual_done}/{expected_done}. "
            "Do not finalize yet; inspect/requeue failed array tasks."
        )

    oracle=read_fragments(root/"oracle")
    val=read_fragments(root/"validation")
    failures=read_fragments(root/"failures")

    oracle.to_csv(RESULTS/f"synthetic_oracle_{args.profile}.csv",index=False)
    val.to_csv(RESULTS/f"synthetic_validation_{args.profile}.csv",index=False)
    failures.to_csv(RESULTS/f"synthetic_failures_{args.profile}.csv",index=False)

    write_manifest(
        RESULTS/f"synthetic_manifest_{args.profile}.json", cfg_all,
        {"models":["logistic","random_forest"],
         "methods":["none","ros","smote","borderline_smote","adasyn","class_weight"],
         "repeats":reps,
         "n_informative":cfg.get("n_informative",5),
         "n_redundant":cfg.get("n_redundant",2),
         "ap_validation_reference":"prevalence-matched oracle AP",
         "checkpointed":True,
         "execution":"ALICE SLURM array by repetition"}
    )

    print(f"Finalized {actual_done}/{expected_done} cells")
    print(f"Oracle rows: {len(oracle):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Failure rows: {len(failures):,}")
    print(f"Next: python3 analyze_results_v2.py --profile {args.profile}")

if __name__ == "__main__":
    main()
