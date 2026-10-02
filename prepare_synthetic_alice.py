from pathlib import Path
import argparse, json
from study_common_alice import METHODS, MODELS
from run_synthetic_alice import _scientific_signature

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--profile", choices=["pilot","full"], default="full")
    p.add_argument("--models", nargs="+", choices=MODELS,
                   default=["logistic","random_forest"])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args=p.parse_args()

    cfg_all=json.loads((ROOT/"config.json").read_text())
    checkpoint_root=RESULTS/"checkpoints"/f"synthetic_{args.profile}"
    signature_path=checkpoint_root/"checkpoint_manifest.json"
    expected=_scientific_signature(cfg_all,args.profile,args.models,args.methods)

    for sub in ("oracle","validation","failures","done"):
        (checkpoint_root/sub).mkdir(parents=True,exist_ok=True)

    if signature_path.exists():
        existing=json.loads(signature_path.read_text())
        if existing != expected:
            raise SystemExit(
                "Existing checkpoint manifest does not match the requested scientific design. "
                "Do not overwrite it; check that you copied the correct project/config."
            )
        print(f"Checkpoint manifest verified: {signature_path}")
    else:
        signature_path.write_text(json.dumps(expected,indent=2,sort_keys=True))
        print(f"Created checkpoint manifest: {signature_path}")

    n_done=len(list((checkpoint_root/"done").glob("*.done")))
    print(f"Existing completed cells: {n_done}")

if __name__ == "__main__":
    main()
