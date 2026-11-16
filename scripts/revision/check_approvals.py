"""
Approval gate for model-written public text (revision plan part 2, section 0).

Prose, captions, legends and figure annotations written by a model stay private
until the authors approve them. Approval is recorded in
revision_private/part2/approvals.json (private, never committed):

    {"approved": [{"path": "...", "sha256": "<hex>", "date": "YYYY-MM-DD", "approver": "..."}]}

A protected file may be committed only if the SHA-256 of its content is listed
there, or if it has a sidecar "<file>.built_from.json" ({"template": "<path>",
"sha256": "<hex>"}) naming a template whose hash is listed (a build from an
approved template). Deterministic outputs (CSVs, stats JSON, figures whose
text is only axis and registry labels) are not protected.

    python scripts/revision/check_approvals.py --staged     # pre-commit hook
    python scripts/revision/check_approvals.py --worktree   # changed or untracked files
    python scripts/revision/check_approvals.py FILE ...

Exit status 1 lists every protected file that is not approved. Only the
authors add entries to approvals.json (or the agent adds an entry they dictate);
never bypass the hook with `git commit --no-verify`.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APPROVALS = os.path.join(ROOT, "revision_private", "part2", "approvals.json")

# Files that carry model-written text (paths relative to the repository root).
PROTECTED = [
    "appendix/*",                                   # public appendix: template, HTML, PDF, figures
    "results_revision/figure_legends*.md",          # figure legends
    "*/figure_legends*.md",
    "*legend*.md",
    "summary_w_figures/revision/examples/*.png",    # example figures with written titles and annotations
    "summary_w_figures/revision/examples/*.pdf",
    "summary_w_figures/revision/part2/*.png",       # part 2 figures with written annotations
    "summary_w_figures/revision/part2/*.pdf",
]
SIDECAR = ".built_from.json"


def is_protected(path: str) -> bool:
    p = path.replace(os.sep, "/")
    if p.endswith(SIDECAR):
        return False
    return any(fnmatch.fnmatch(p, pat) for pat in PROTECTED)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_approved(path: str = APPROVALS) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return {e["sha256"].lower() for e in data.get("approved", []) if e.get("sha256")}


def _read(path: str, staged: bool) -> bytes | None:
    if staged:
        r = subprocess.run(["git", "show", f":{path}"], cwd=ROOT, capture_output=True)
        return r.stdout if r.returncode == 0 else None
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def unapproved(paths, approved: set[str], staged: bool = False) -> list[str]:
    """Protected paths among `paths` whose content is not approved. Deleted
    files are allowed (removing text needs no approval)."""
    bad = []
    for p in paths:
        if not is_protected(p):
            continue
        data = _read(p, staged)
        if data is None:
            continue
        if sha256(data) in approved:
            continue
        side = _read(p + SIDECAR, staged)
        if side is not None:
            try:
                t = json.loads(side)
                if t.get("sha256", "").lower() in approved:
                    continue
            except ValueError:
                pass
        bad.append(p)
    return bad


def _git_list(args) -> list[str]:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()]


def staged_paths() -> list[str]:
    return _git_list(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def worktree_paths() -> list[str]:
    changed = _git_list(["diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
    untracked = _git_list(["ls-files", "--others", "--exclude-standard"])
    return sorted(set(changed) | set(untracked))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--worktree", action="store_true")
    a = ap.parse_args(argv)
    approved = load_approved()
    if a.staged:
        bad = unapproved(staged_paths(), approved, staged=True)
    elif a.worktree:
        bad = unapproved(worktree_paths(), approved)
    else:
        bad = unapproved(a.files, approved)
    if bad:
        print("Not approved (model-written text needs author approval; see revision plan part 2, section 0):")
        for p in bad:
            print(f"  {p}")
        print("Record the approved content's SHA-256 in revision_private/part2/approvals.json "
              "(authors only), or keep the file private. Do not use --no-verify.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
