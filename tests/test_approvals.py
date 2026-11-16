"""Approval gate for model-written public text (scripts/revision/check_approvals.py)."""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.revision import check_approvals as ca  # noqa: E402


def test_protected_patterns():
    assert ca.is_protected("appendix/appendix.pdf")
    assert ca.is_protected("results_revision/figure_legends.md")
    assert ca.is_protected("summary_w_figures/revision/examples/E2a_fig11_grounding_vignettes_score_view_grid.png")
    assert ca.is_protected("summary_w_figures/revision/part2/fig_safety.pdf")
    assert not ca.is_protected("results_revision/distribution_summary.csv")
    assert not ca.is_protected("summary_w_figures/revision/examples/E3_ranking_shifts_data.csv")
    assert not ca.is_protected("appendix/fig.png" + ca.SIDECAR)


def test_unapproved_blocked_and_approved_passes():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "appendix"))
        f = os.path.join(d, "appendix", "t.html")
        open(f, "w").write("<p>draft</p>")
        old = ca.ROOT
        try:
            ca.ROOT = d
            assert ca.unapproved(["appendix/t.html"], set()) == ["appendix/t.html"]
            h = ca.sha256(open(f, "rb").read())
            assert ca.unapproved(["appendix/t.html"], {h}) == []
            # built from an approved template
            open(f, "w").write("<p>built 1.23</p>")
            open(f + ca.SIDECAR, "w").write(json.dumps({"template": "x", "sha256": h}))
            assert ca.unapproved(["appendix/t.html"], {h}) == []
            # deleted files are allowed
            assert ca.unapproved(["appendix/missing.html"], set()) == []
        finally:
            ca.ROOT = old


def test_worktree_changes_are_approved():
    """Any protected file changed or added in the working tree must be approved
    before it can be committed (the pre-commit hook enforces the same rule)."""
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        return
    bad = ca.unapproved(ca.worktree_paths(), ca.load_approved())
    assert not bad, f"unapproved model-written files in the working tree: {bad}"
