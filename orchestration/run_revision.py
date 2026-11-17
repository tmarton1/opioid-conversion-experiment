"""
Rebuild everything the JMIR revision analysis has produced so far, in order.

    python orchestration/run_revision.py                 # all steps
    python orchestration/run_revision.py --steps rescore,references
    python orchestration/run_revision.py --skip-figures  # CSVs only

Steps (each overwrites its own outputs; safe to rerun):
  rescore      analysis/revision/rescore.py → results_<score>/, results_revision/responses_long.csv
  pipeline     for every registered score: orchestration/run_analysis.py with
               RESULTS_DIR=results_<score> (Nielsen: the raw CSVs copied into
               results_nielsen/, empty-vignette prompts scored against the
               conversion they ask for), then generate_summaries.py into figures_<score>/
  references   analysis/revision/reference_comparison.py → results_revision/references/
  bands        analysis/revision/tolerances.py → results_revision/references/band_justification*.csv
  analyses     Phase 3 modules (PHASE3_MODULES) and the clinical-reduction check
               → results_revision/*.csv
  report       example figures E1-E8 and the private differences report
               (orchestration/build_differences_report.py)

Figures for any score/view/bands: python -m orchestration.build_figures (Phase 2).

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
            from analysis.revision.vignette_targets import VIGNETTE_FILE, load_raw
            for f in RAW_FILES:
                if f == VIGNETTE_FILE:
                    # empty-vignette prompts scored against the conversion they ask for
                    df = load_raw(RAW_DIR, f)
                    tmp = os.path.join(rdir, f + ".tmp")
                    df.to_csv(tmp, index=False)
                    os.replace(tmp, os.path.join(rdir, f))
                else:
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


def step_bands(args):
    _run([PY, "-m", "analysis.revision.tolerances"])


# Phase 3 analyses (each module writes its own CSVs to results_revision/),
# plus the clinical-reduction check for the vignette decisions.
PHASE3_MODULES = [
    "analysis.revision.vignette_reductions",
    "analysis.revision.distribution_metrics",   # 3A
    "analysis.revision.circularity",            # 3B
    "analysis.revision.extreme_breakdown",      # 3A follow-up: extreme-error classes, per-model tail tests
    "analysis.revision.range_rates_by_prompt",  # range measures by prompt type on comparable items
    "analysis.revision.error_taxonomy",         # FA.4: error taxonomy (outside broad, both directions)
    "analysis.revision.rankings_by_criterion",  # FA.4: rankings by criterion and rank correlations
    "analysis.revision.multiple_comparisons",   # decision 45: Bonferroni within each family, rankings on 0.6-1.7x only
    "analysis.revision.di2_subset",             # 3C
    "analysis.revision.model_agreement",        # 3D
    "analysis.revision.standard_matches",       # 3E
    "analysis.revision.units_vignette_split",   # 3F
]


def step_analyses(args):
    for mod in PHASE3_MODULES:
        path = os.path.join(*mod.split(".")) + ".py"
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing")
        _run([PY, "-m", mod])


def step_report(args):
    # differences report: public script, private template/output (skipped with a
    # message when revision_private/ is absent); regenerates figures E1-E7
    if not os.path.isdir("revision_private"):
        print("revision_private/ absent: building the example figures only")
        for mod in ("visualize.revision.examples_scores", "visualize.revision.examples_references",
                    "visualize.revision.examples_safety"):
            _run([PY, "-m", mod])
        return
    _run([PY, "orchestration/build_differences_report.py"])


STEPS = {
    "rescore": step_rescore,
    "pipeline": step_pipeline,
    "references": step_references,
    "bands": step_bands,
    "analyses": step_analyses,
    "report": step_report,
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
