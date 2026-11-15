"""
Build any selection of figures under any scores, views and bands.

    python -m orchestration.build_figures --figures 1,11 --scores nielsen,all,non_nielsen \
        --views bands,distribution --compare
    python -m orchestration.build_figures --figures all --scores nielsen --views bands   # reproduction
    python -m orchestration.build_figures --figures 11 --scores nielsen,non_nielsen \
        --views bands --bands broad --compare
    python -m orchestration.build_figures --list

Scores, bands, figures and conditions come from their registries
(analysis/revision/scores.py, tolerances.py, conditions.py,
visualize/revision/registry.py). Score-independent figures (4-7) are built
from Nielsen only, bands view only; asking for them under other scores or the
distribution view skips them with a note. Output goes to
$FIGURES_DIR/revision/ (or --out); paths encode figure, scores, view and bands.
Never queries any model.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.revision import style  # noqa: E402,F401  (matplotlib backend and styling)
from analysis.revision.scores import SCORES  # noqa: E402
from analysis.revision.tolerances import BANDS, DEFAULT_BANDS  # noqa: E402
from visualize.revision import compose  # noqa: E402
from visualize.revision.registry import FIGURES, parse_selection  # noqa: E402


def _list():
    print("Figures:")
    for f in FIGURES.values():
        print(f"  {f.num:2d}  {f.slug:32s} rescorable={f.rescorable!s:5s} views={','.join(f.views)}")
    print("Scores:", ", ".join(SCORES))
    print("Bands: ", ", ".join(BANDS), f"(default {','.join(DEFAULT_BANDS)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--figures", default="all")
    ap.add_argument("--scores", default="nielsen")
    ap.add_argument("--views", default="bands")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--compare", action="store_true", help="one figure comparing the scores side by side")
    ap.add_argument("--out", default=None, help="output root (default $FIGURES_DIR/revision)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        return _list()
    scores = [s for s in a.scores.split(",") if s]
    views = [v for v in a.views.split(",") if v]
    bands = [b for b in a.bands.split(",") if b]
    for s in scores:
        if s not in SCORES:
            raise SystemExit(f"unknown score {s!r}; registered: {list(SCORES)}")
    for b in bands:
        if b not in BANDS:
            raise SystemExit(f"unknown band {b!r}; registered: {list(BANDS)}")
    built, skipped = [], []
    for num in parse_selection(a.figures):
        spec = FIGURES[num]
        for view in views:
            if view not in spec.views:
                skipped.append(f"figure {num} ({view}): score-independent, bands view only")
                continue
            if not spec.rescorable and scores != ["nielsen"]:
                skipped.append(f"figure {num}: score-independent, built from Nielsen only")
            try:
                for paths in compose.build(num, scores, view, bands, a.compare, a.out):
                    built.append(paths[0])
                    print("✓", paths[0], flush=True)
            except Exception as e:  # report and continue with the rest of the selection
                skipped.append(f"figure {num} ({view}): {type(e).__name__}: {e}")
                print("✗", skipped[-1], flush=True)
    print(f"\n{len(built)} figure(s) built.")
    for s in skipped:
        print("  note:", s)
    return 0 if not any(":" in s and "Error" in s for s in skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
