"""
Approval gate for model-written public text (revision plan part 2, section 0).

Prose, captions, legends and figure annotations written by a model stay private
until the authors approve them. Approval is recorded in
revision_private/part2/approvals.json (private, never committed):

    {"approved": [{"path": "...", "sha256": "<hex>", "status": "approved" | "accepted_unreviewed",
                   "date": "YYYY-MM-DD", "approver": "...", "note": "..."}]}

Statuses:
  approved             reviewed and approved by the authors;
  accepted_unreviewed  accepted for use without author review (the part 1 legends
                       and example figures E1-E8). Allowed on the branch, in the
                       private differences report and the example figures, but
                       never in the public appendix.
A protected file may be committed only while the SHA-256 of its content is
listed (any change needs a new "approved" entry), or if it has a sidecar
"<file>.built_from.json" ({"template": "<path>", "sha256": "<hex>"}) naming a
template whose hash is listed (a build from that template; the build gets the
template's status). Files under appendix/ need status "approved".
Deterministic outputs (CSVs, stats JSON, figures whose text is only axis and
registry labels) are not protected.

    python scripts/revision/check_approvals.py --staged     # pre-commit hook
    python scripts/revision/check_approvals.py --worktree   # changed or untracked files
    python scripts/revision/check_approvals.py FILE ...
    python scripts/revision/check_approvals.py --list       # recorded entries by status

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


APPROVED, ACCEPTED = "approved", "accepted_unreviewed"
STATUSES = (APPROVED, ACCEPTED)
APPENDIX_ONLY_APPROVED = ["appendix/*"]


def load_entries(path: str = APPROVALS) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    out = []
    for e in data.get("approved", []):
        if not e.get("sha256"):
            continue
        st = e.get("status", APPROVED)
        if st not in STATUSES:
            raise ValueError(f"unknown approval status {st!r} for {e.get('path')}")
        out.append({**e, "sha256": e["sha256"].lower(), "status": st})
    return out


def load_approved(path: str = APPROVALS) -> dict[str, str]:
    """sha256 -> status ("approved" wins if a hash is recorded with both)."""
    out: dict[str, str] = {}
    for e in load_entries(path):
        if out.get(e["sha256"]) != APPROVED:
            out[e["sha256"]] = e["status"]
    return out


def _as_status_map(approved) -> dict[str, str]:
    if isinstance(approved, dict):
        return {k.lower(): v for k, v in approved.items()}
    return {h.lower(): APPROVED for h in approved}      # a plain set means "approved"


def needs_approved(path: str) -> bool:
    p = path.replace(os.sep, "/")
    return any(fnmatch.fnmatch(p, pat) for pat in APPENDIX_ONLY_APPROVED)


def _read(path: str, staged: bool) -> bytes | None:
    if staged:
        r = subprocess.run(["git", "show", f":{path}"], cwd=ROOT, capture_output=True)
        return r.stdout if r.returncode == 0 else None
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def status_of(path: str, approved, staged: bool = False):
    """Status of a protected file's current content: "approved",
    "accepted_unreviewed", or None (not recorded). Unprotected or deleted -> "free"."""
    if not is_protected(path):
        return "free"
    smap = _as_status_map(approved)
    data = _read(path, staged)
    if data is None:
        return "free"
    st = smap.get(sha256(data))
    if st is None:
        side = _read(path + SIDECAR, staged)
        if side is not None:
            try:
                st = smap.get(json.loads(side).get("sha256", "").lower())
            except ValueError:
                st = None
    return st


def classify(paths, approved, staged: bool = False) -> dict[str, list[str]]:
    """{"blocked": [...], "approved": [...], "accepted_unreviewed": [...]} for the
    protected paths among `paths`."""
    out = {"blocked": [], APPROVED: [], ACCEPTED: []}
    for p in paths:
        st = status_of(p, approved, staged)
        if st == "free":
            continue
        if st is None or (st == ACCEPTED and needs_approved(p)):
            out["blocked"].append(p)
        else:
            out[st].append(p)
    return out


def unapproved(paths, approved, staged: bool = False) -> list[str]:
    """Protected paths among `paths` that may not be committed."""
    return classify(paths, approved, staged)["blocked"]


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
    ap.add_argument("--list", action="store_true", help="list recorded entries by status")
    a = ap.parse_args(argv)
    if a.list:
        entries = load_entries()
        for st in STATUSES:
            rows = [e for e in entries if e["status"] == st]
            print(f"{st} ({len(rows)}):")
            for e in rows:
                print(f"  {e.get('path')}  {e['sha256'][:12]}  {e.get('date', '')}  {e.get('approver', '')}  {e.get('note', '')}")
        return 0
    approved = load_approved()
    if a.staged:
        res = classify(staged_paths(), approved, staged=True)
    elif a.worktree:
        res = classify(worktree_paths(), approved)
    else:
        res = classify(a.files, approved)
    for st in (APPROVED, ACCEPTED):
        if res[st]:
            print(f"{st} ({len(res[st])}):")
            for p in res[st]:
                print(f"  {p}")
    if res["blocked"]:
        print("Not approved (model-written text needs author approval; see revision plan part 2, section 0;"
              " files under appendix/ need status \"approved\"):")
        for p in res["blocked"]:
            print(f"  {p}")
        print("Record the approved content's SHA-256 in revision_private/part2/approvals.json "
              "(authors only), or keep the file private. Do not use --no-verify.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
