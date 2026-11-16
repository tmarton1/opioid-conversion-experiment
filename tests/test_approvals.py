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
    assert ca.is_protected("summary_w_figures/revision/figures_appendix/fig_safety.pdf")
    assert ca.is_protected("figures_revision/main/fig_circularity.png")
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


def test_two_statuses_and_appendix_rule():
    """accepted_unreviewed passes outside appendix/ only while the content is
    unchanged; appendix/ needs "approved"; any change needs a new entry."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "results_revision"))
        os.makedirs(os.path.join(d, "appendix"))
        leg = os.path.join(d, "results_revision", "figure_legends_X.md")
        app = os.path.join(d, "appendix", "figure_legends_X.md")
        open(leg, "w").write("Accepted from part 1 without author review.\n\nlegend")
        open(app, "w").write("Accepted from part 1 without author review.\n\nlegend")
        h = ca.sha256(open(leg, "rb").read())
        old = ca.ROOT
        try:
            ca.ROOT = d
            acc = {h: ca.ACCEPTED}
            res = ca.classify(["results_revision/figure_legends_X.md", "appendix/figure_legends_X.md"], acc)
            assert res[ca.ACCEPTED] == ["results_revision/figure_legends_X.md"]
            assert res["blocked"] == ["appendix/figure_legends_X.md"]   # same content, but appendix needs approved
            res = ca.classify(["appendix/figure_legends_X.md"], {h: ca.APPROVED})
            assert res[ca.APPROVED] == ["appendix/figure_legends_X.md"] and not res["blocked"]
            # approved wins when a hash is recorded with both statuses
            f = os.path.join(d, "a.json")
            json.dump({"approved": [{"path": "x", "sha256": h, "status": ca.ACCEPTED},
                                    {"path": "x", "sha256": h, "status": ca.APPROVED}]}, open(f, "w"))
            assert ca.load_approved(f) == {h: ca.APPROVED}
            # a changed file is no longer accepted
            open(leg, "a").write(" edited")
            assert ca.unapproved(["results_revision/figure_legends_X.md"], acc) == ["results_revision/figure_legends_X.md"]
        finally:
            ca.ROOT = old


def test_unknown_status_rejected():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "a.json")
        json.dump({"approved": [{"path": "x", "sha256": "ab", "status": "maybe"}]}, open(f, "w"))
        try:
            ca.load_entries(f)
        except ValueError:
            return
        raise AssertionError("unknown status accepted")
