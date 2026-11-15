"""
Rebuild everything the JMIR revision analysis has produced so far, in order.

    python orchestration/run_revision.py                 # all steps
    python orchestration/run_revision.py --steps rescore,references
    python orchestration/run_revision.py --skip-figures  # CSVs only

Steps (each overwrites its own outputs; safe to rerun):
  rescore      analysis/revision/rescore.py → results_<score>/, results_revision/responses_long.csv
  pipeline     for every registered score: orchestration/run_analysis.py with
               RESULTS_DIR=results_<score> (Nielsen: the raw CSVs copied into
               results_nielsen/), then generate_summaries.py into figures_<score>/
  references   analysis/revision/reference_comparison.py → results_revision/references/

Never calls any model API. Writes results_revision/run_log.json with the git
commit, timestamp and the SHA-256 of every reference JSON file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from analysis.revision.scores import SCORES  # noqa: E402
from resources.reference_data import load_index  # noqa: E402

RAW_DIR = "results"
RAW_FILES = ["experiment_results.csv", "explain_units_experiment_results.csv",
             "reworded_vignettes_results.csv"]
PY = sys.executable


def _env(score: str) -> dict:
    env = dict(os.environ)
    env.update(RESULTS_DIR=f"results_{score}", FIGURES_DIR=f"figures_{score}",
               RAW_FIGURES_DIR=f"raw_figures_{score}")
    return env


def _run(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def step_rescore(args):
    _run([PY, "-m", "analysis.revision.rescore", "--raw-dir", RAW_DIR])


def step_pipeline(args):
    for score in SCORES:
        if args.scores and score not in args.scores:
            continue
        rdir = f"results_{score}"
        if score == "nielsen":
            os.makedirs(rdir, exist_ok=True)
            for f in RAW_FILES:
                shutil.copy2(os.path.join(RAW_DIR, f), os.path.join(rdir, f))
        elif not os.path.exists(os.path.join(rdir, RAW_FILES[0])):
            raise SystemExit(f"{rdir}/ missing: run the rescore step first")
        _run([PY, "orchestration/run_analysis.py"], env=_env(score))
        if not args.skip_figures:
            _run([PY, "orchestration/generate_summaries.py"], env=_env(score))
        else:
            _run([PY, "orchestration/extract_figure_stats.py"], env=_env(score))


def step_references(args):
    _run([PY, "-m", "analysis.revision.reference_comparison", "--raw-dir", RAW_DIR])


STEPS = {
    "rescore": step_rescore,
    "pipeline": step_pipeline,
    "references": step_references,
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_log(steps):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    refs = {name: {"file": e["file"], "sha256": _sha256(e["file"])} for name, e in load_index().items()}
    log = {
        "git_commit": commit, "working_tree_modified": dirty,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "steps": steps, "scores": list(SCORES), "reference_files": refs,
        "random_seeds": {},
    }
    os.makedirs("results_revision", exist_ok=True)
    tmp = "results_revision/run_log.json.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, "results_revision/run_log.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the revision outputs.")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma list from {list(STEPS)}")
    ap.add_argument("--scores", default="", help="limit the pipeline step to these scores")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args(argv)
    args.scores = [s for s in args.scores.split(",") if s]
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; choose from {list(STEPS)}")
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        STEPS[s](args)
    write_log(steps)
    print("\n✓ run_revision complete")


if __name__ == "__main__":
    main()
