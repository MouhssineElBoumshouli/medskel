"""Figure 9: is the inter-observer result robust, or is it a few lucky images?

Experiment 08 reports medians and Wilcoxon p-values over 28 images. That is the
headline claim, so it deserves the checks a sceptical reader would apply before
believing it. Three of them here, and any one can sink the result:

1. Sign test. Forget magnitudes: on how many of the 28 images does the bisector
   method win at all? A median can be dragged by a handful of images. If the
   direction only holds on 15 of 28, the median is not describing the dataset.

2. Leave-one-out. Recompute the p-value 28 times, each time dropping a
   different image. If any single image can move it across 0.05, the result is
   one image away from not existing.

3. Tolerance sensitivity. Skeleton agreement counts a point as matching if it
   lands within 2px of the other observer's skeleton. 2px was my choice. If the
   direction flips at 1px or 5px, the finding is about the threshold rather
   than about the methods.

None of these can prove the result. They can only fail to break it, which is
the most any of this gets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, binomtest

from medskel import chase, viz, baseline
from medskel.voronoi import skeletonize
from medskel.metrics import skeleton_agreement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

MEASURES = ["total_length", "n_bifurcations", "median_calibre",
            "median_tortuosity", "agreement", "agreement_core"]
HIGHER_IS_BETTER = {"agreement", "agreement_core"}
TOLERANCES = [1.0, 2.0, 3.0, 5.0]


def load_pairs(records, measure, a="thinning", b="bisector eps=4"):
    va = np.array([r[f"{a}|{measure}"] for r in records], float)
    vb = np.array([r[f"{b}|{measure}"] for r in records], float)
    ok = np.isfinite(va) & np.isfinite(vb)
    return va[ok], vb[ok]


def sign_and_loo(records):
    print("1. SIGN TEST and 2. LEAVE-ONE-OUT  (bisector eps=4 vs thinning)\n")
    print(f"{'measure':<20}{'bisector wins':>15}{'sign p':>10}"
          f"{'LOO p range':>22}{'verdict':>12}")
    out = {}
    for m in MEASURES:
        va, vb = load_pairs(records, m)
        n = len(va)
        # "wins" means better, which flips for the agreement measures
        wins = int(np.sum(vb > va)) if m in HIGHER_IS_BETTER else int(np.sum(vb < va))
        sign_p = binomtest(wins, n, 0.5).pvalue

        ps = []
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            if np.allclose(va[keep], vb[keep]):
                ps.append(1.0)
            else:
                ps.append(wilcoxon(va[keep], vb[keep]).pvalue)
        lo, hi = min(ps), max(ps)
        stable = "stable" if hi < 0.05 else ("fragile" if lo < 0.05 else "null")

        print(f"{m:<20}{f'{wins}/{n}':>15}{sign_p:>10.4f}"
              f"{f'{lo:.4f} - {hi:.4f}':>22}{stable:>12}")
        out[m] = {"wins": wins, "n": n, "sign_p": float(sign_p),
                  "loo_p_min": float(lo), "loo_p_max": float(hi),
                  "verdict": stable}
    return out


def tolerance_sweep(cases):
    print("\n3. TOLERANCE SENSITIVITY of skeleton agreement")
    print("   (higher is better; does thinning keep winning at every radius?)\n")
    print(f"{'tolerance':<12}{'thinning':>12}{'bisector e=4':>15}{'winner':>12}")

    per_tol = {t: {"thinning": [], "bisector": []} for t in TOLERANCES}
    for c in cases:
        st1, st2 = baseline.thinning(c["obs1"]), baseline.thinning(c["obs2"])
        sb1 = skeletonize(c["obs1"], epsilon=4.0, prune=1.0)
        sb2 = skeletonize(c["obs2"], epsilon=4.0, prune=1.0)
        for t in TOLERANCES:
            per_tol[t]["thinning"].append(skeleton_agreement(st1, st2, tol=t))
            per_tol[t]["bisector"].append(skeleton_agreement(sb1, sb2, tol=t))

    out = {}
    for t in TOLERANCES:
        a = float(np.median(per_tol[t]["thinning"]))
        b = float(np.median(per_tol[t]["bisector"]))
        winner = "thinning" if a > b else "bisector"
        print(f"{t:<12.1f}{a:>12.3f}{b:>15.3f}{winner:>12}")
        out[str(t)] = {"thinning": a, "bisector": b, "winner": winner}
    return out, per_tol


def figure(records, per_tol):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    va, vb = load_pairs(records, "total_length")
    top = max(va.max(), vb.max())
    ax.plot([0, top], [0, top], "-", color="grey", lw=1)
    ax.plot(va, vb, "o", color="#d62728", ms=6)
    ax.set_xlabel("thinning: change between observers")
    ax.set_ylabel("bisector: change between observers")
    ax.set_title(f"total vessel length, one dot per image\n"
                 f"{int(np.sum(vb < va))} of {len(va)} below the line "
                 f"(bisector better)", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    va, vb = load_pairs(records, "agreement")
    lim = [min(va.min(), vb.min()) - .02, max(va.max(), vb.max()) + .02]
    ax.plot(lim, lim, "-", color="grey", lw=1)
    ax.plot(va, vb, "o", color="#1f77b4", ms=6)
    ax.set_xlabel("thinning: skeleton agreement")
    ax.set_ylabel("bisector: skeleton agreement")
    ax.set_title(f"positional agreement\n"
                 f"{int(np.sum(vb > va))} of {len(va)} above the line "
                 f"(bisector better)", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for name, col in [("thinning", "#ff7f0e"), ("bisector", "#d62728")]:
        ax.plot(TOLERANCES, [np.median(per_tol[t][name]) for t in TOLERANCES],
                "-o", color=col, label=name)
    ax.set_xlabel("matching tolerance (px)")
    ax.set_ylabel("median skeleton agreement")
    ax.set_title("does the answer depend on my\nchoice of 2px?", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    viz.finish(fig, os.path.join(OUT, "09_robustness.png"))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "interobserver.json")
    if not os.path.exists(path):
        print("run experiments/08_interobserver.py first")
        return 1
    records = json.load(open(path))

    summary = {"sign_and_loo": sign_and_loo(records)}

    try:
        cases = chase.load_cases()
    except chase.ChaseNotAvailable as e:
        print(f"\n(skipping tolerance sweep: {e})")
        cases = None

    if cases:
        summary["tolerance"], per_tol = tolerance_sweep(cases)
        figure(records, per_tol)

    with open(os.path.join(RESULTS, "robustness.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\nA result survives this if every headline row says 'stable' and the")
    print("tolerance winner never flips. Anything else is worth saying out loud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
