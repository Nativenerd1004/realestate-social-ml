"""
main.py — Master Orchestrator
Run this once to execute the full pipeline from Kaggle → Dashboard → Cloudflare.

Usage:
    python main.py                  # full pipeline
    python main.py --from 5         # resume from step 5
    python main.py --only 9         # run a single step
    python main.py --no-deploy      # skip GitHub/Cloudflare
"""
import argparse
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

LOG_PATH = Path(__file__).parent.parent / "pipeline.log"

def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    sys.stdout.reconfigure(line_buffering=True)  # flush every line


STEPS = {
    1:  ("Kaggle Dataset Fetch",          "steps.01_kaggle_fetch",   "run"),
    2:  ("PostgreSQL Load",               "steps.02_postgres_load",  "run"),
    3:  ("Data Cleaning",                 "steps.03_clean",          "run"),
    4:  ("Feature Engineering",           "steps.04_features",       "run"),
    5:  ("Model Training",                "steps.05_train_models",   "run"),
    6:  ("Evaluate + Visualize",          "steps.06_evaluate",       "run"),
    7:  ("Auto-Retrain Loop",             "steps.07_retrain_loop",   "run"),
    8:  ("Notebook Generation",           "steps.08_notebook_gen",   "run"),
    9:  ("Dashboard Build",               "steps.09_dashboard",      "run"),
    10: ("GitHub + Cloudflare Deploy",    "steps.10_deploy",         "run"),
}


def banner():
    print("\n" + "═" * 62)
    print("  REAL ESTATE + SOCIAL MEDIA — ML PIPELINE AGENT")
    print("  Samuel's End-to-End Autonomous ML System")
    print("═" * 62 + "\n")


def run_step(step_num: int):
    name, module_path, func_name = STEPS[step_num]
    print(f"\n{'━'*62}")
    print(f"  STEP {step_num:02d} / 10 — {name}")
    print(f"{'━'*62}")
    t0 = time.time()

    import importlib
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, func_name)
    result = fn()

    elapsed = time.time() - t0
    print(f"\n  ✓ Done in {elapsed:.1f}s")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_step", type=int, default=1,
                   help="Start from this step number (default: 1)")
    p.add_argument("--only", dest="only_step", type=int, default=None,
                   help="Run only this step")
    p.add_argument("--no-deploy", action="store_true",
                   help="Skip Step 10 (GitHub + Cloudflare)")
    return p.parse_args()


def main():
    setup_logging()
    banner()
    args = parse_args()

    if args.only_step:
        steps = [args.only_step]
    else:
        steps = list(range(args.from_step, 11))
        if args.no_deploy and 10 in steps:
            steps.remove(10)

    results = {}
    for step in steps:
        if step not in STEPS:
            print(f"[!] Step {step} does not exist (valid: 1–10)")
            continue
        try:
            results[step] = run_step(step)
        except SystemExit:
            print(f"\n[✗] Step {step} exited early — check output above")
            sys.exit(1)
        except Exception as e:
            print(f"\n[✗] Step {step} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "═" * 62)
    print("  PIPELINE COMPLETE")
    print("═" * 62)

    if 9 in steps:
        from config import DASHBOARD
        print(f"\n  Dashboard : file://{DASHBOARD}/index.html")
    if 10 in steps:
        from config import BASE_DIR
        import json
        dep = BASE_DIR / "DEPLOYMENT.json"
        if dep.exists():
            d = json.loads(dep.read_text())
            if d.get("github_url"):
                print(f"  GitHub    : {d['github_url']}")
            if d.get("cloudflare_url"):
                print(f"  Live URL  : {d['cloudflare_url']}")

    print()


if __name__ == "__main__":
    main()
