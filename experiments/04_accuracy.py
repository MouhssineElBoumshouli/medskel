"""Figure 4 + table: accuracy against phantoms with a known centerline.

Noise robustness is worth nothing if the skeleton is in the wrong place, so
this is the other half of the check. Clean phantoms, known answer, four
numbers per method:

  mean / p95 error  distance from the skeleton to the true centerline
  coverage          fraction of the true centerline that got found at all,
                    which is what stops "return almost nothing" from winning
  branches          against the number the phantom actually has
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import matplotlib
matplotlib.use("Agg")
import numpy as np

from medskel import phantoms, viz, baseline
from medskel.voronoi import skeletonize
from medskel.metrics import centerline_error, coverage, centeredness, compactness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

CASES = [
    ("straight tube", phantoms.straight_tube, 1),
    ("curved tube", phantoms.curved_tube, 1),
    ("tapered tube", phantoms.tapered_tube, 1),
    ("bifurcation", phantoms.bifurcation, 3),
    ("vessel tree", phantoms.vessel_tree, 31),
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    panels = []

    for name, fn, expected in CASES:
        mask, truth = fn()
        print(f"\n{name}  ({mask.sum()} px, expected {expected} branches)")

        t0 = time.time()
        thin = baseline.thinning(mask)
        t_thin = time.time() - t0
        c = baseline.count_pixel_branches(thin)
        rows.append(_row(name, "thinning", thin, truth, mask, c["branches"],
                         expected, t_thin, None))

        t0 = time.time()
        sk = skeletonize(mask, epsilon=2.0, spacing=1.0, prune=1.0)
        t_ours = time.time() - t0
        rows.append(_row(name, "ours eps=2", sk, truth, mask, sk.n_branches(),
                         expected, t_ours, compactness(sk, thin)))

        panels.append((name, mask, truth, sk, thin))
        for r in rows[-2:]:
            print(f"  {r['method']:12s} err={r['mean_error']:.2f}/"
                  f"{r['p95_error']:.2f}px  cover={r['coverage']:.2f}  "
                  f"branches={r['branches']} (want {expected})  "
                  f"{r['seconds']:.2f}s")

    with open(os.path.join(RESULTS, "phantom_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=1)

    _figure(panels)
    _markdown(rows)


def _row(case, method, estimate, truth, mask, branches, expected, seconds, comp):
    e = centerline_error(estimate, truth)
    ctr = centeredness(estimate, mask)
    return {
        "case": case, "method": method,
        "mean_error": e["mean"], "p95_error": e["p95"],
        "coverage": coverage(estimate, truth, tol=3.0),
        "centeredness": ctr["mean_ratio"],
        "branches": int(branches), "expected_branches": int(expected),
        "seconds": float(seconds),
        "polyline_vertices": comp["polyline_vertices"] if comp else None,
        "skeleton_pixels": comp["skeleton_pixels"] if comp else None,
    }


def _figure(panels):
    fig, axes = viz.panel_grid(len(panels), ncols=len(panels), size=4.2,
                               aspect=0.95)
    for ax, (name, mask, truth, sk, thin) in zip(axes, panels):
        viz.show_mask(ax, mask, cmap="gray")
        viz.draw_pixel_skeleton(ax, thin, ms=1.2, label="thinning")
        viz.draw_skeleton(ax, sk, lw=1.5, label="ours")
        viz.draw_centerline(ax, truth, lw=1.2, label="truth")
        ax.set_title(f"{name}\n{sk.n_branches()} branches (ours)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    viz.finish(fig, os.path.join(OUT, "04_accuracy.png"))


def _markdown(rows):
    """Print the table in the shape the README wants it."""
    print("\n\n| phantom | method | mean err | p95 err | coverage | branches "
          "(true) | time |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r['method']} | {r['mean_error']:.2f} px | "
              f"{r['p95_error']:.2f} px | {r['coverage']*100:.0f}% | "
              f"{r['branches']} ({r['expected_branches']}) | "
              f"{r['seconds']:.2f} s |")

    ours = [r for r in rows if r["method"].startswith("ours")]
    print("\ncompactness (polyline vertices vs skeleton pixels)")
    for r in ours:
        if r["polyline_vertices"]:
            print(f"  {r['case']:15s} {r['polyline_vertices']:6d} vs "
                  f"{r['skeleton_pixels']:6d} px  "
                  f"({r['skeleton_pixels']/r['polyline_vertices']:.1f}x)")


if __name__ == "__main__":
    main()
