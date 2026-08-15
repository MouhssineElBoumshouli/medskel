"""Figure 3: what boundary noise does to each method.

This is the experiment the whole project rests on, so it is worth being clear
about what it asks. A segmentation of a vessel is never clean: the mask edge
wobbles by a pixel or two because the vesselness filter is noisy, because the
image is blurry, because a human traced it by hand. None of that wobble is
anatomy, and none of it should turn into a vessel branch.

So: take a phantom whose answer is known (a Y has exactly 3 branches and 1
bifurcation), corrupt only its boundary, and count how many branches each
method reports. Every branch past 3 is an artefact of the noise.

The claim being tested is that doing the smoothing *before* the skeleton exists
(by simplifying the polygon) beats doing it after (by pruning spurs off a pixel
skeleton), because once a spurious branch is in the skeleton you can only
remove it with a length threshold that has no idea what scale it is at.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medskel import phantoms, viz, baseline
from medskel.voronoi import skeletonize
from medskel.metrics import centerline_error, coverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

AMPLITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SEEDS = range(6)
EXPECTED_BRANCHES = 3          # a Y
EXPECTED_ENDPOINTS = 3


def evaluate(mask, truth):
    """Every method's branch count and centerline error on one noisy mask."""
    row = {}

    thin = baseline.thinning(mask)
    counts = baseline.count_pixel_branches(thin)
    row["thinning"] = dict(branches=counts["branches"],
                           endpoints=counts["endpoints"],
                           **_err(thin, truth))

    for min_len in (5, 15):
        pruned = baseline.prune_pixel_skeleton(thin, min_length=min_len)
        c = baseline.count_pixel_branches(pruned)
        row[f"thinning+prune{min_len}"] = dict(branches=c["branches"],
                                               endpoints=c["endpoints"],
                                               **_err(pruned, truth))

    for eps in (1.0, 2.0, 4.0):
        sk = skeletonize(mask, epsilon=eps, spacing=1.0, prune=1.0)
        row[f"bisector eps={eps:g}"] = dict(branches=sk.n_branches(),
                                        endpoints=sk.n_endpoints(),
                                        **_err(sk, truth))
    return row


def _err(estimate, truth):
    e = centerline_error(estimate, truth)
    return {"mean_error": e["mean"], "p95_error": e["p95"],
            "coverage": coverage(estimate, truth, tol=3.0)}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    clean, truth = phantoms.bifurcation()

    records = []
    for amp in AMPLITUDES:
        for seed in SEEDS:
            mask = clean if amp == 0 else phantoms.roughen(clean, amplitude=amp,
                                                           scale=3.0, seed=seed)
            mask = phantoms.largest_component(mask)
            for method, vals in evaluate(mask, truth).items():
                records.append(dict(amplitude=amp, seed=seed, method=method,
                                    **vals))
            if amp == 0:
                break          # nothing random about the clean case
        print(f"  amplitude {amp} done")

    with open(os.path.join(RESULTS, "noise_robustness.json"), "w") as f:
        json.dump(records, f, indent=1)

    methods = ["thinning", "thinning+prune5", "thinning+prune15",
               "bisector eps=1", "bisector eps=2", "bisector eps=4"]
    colors = ["#ff7f0e", "#ffbb78", "#8c564b",
              "#aec7e8", "#1f77b4", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for m, c in zip(methods, colors):
        amps, mean, lo, hi, err = [], [], [], [], []
        for a in AMPLITUDES:
            vals = [r["branches"] for r in records
                    if r["method"] == m and r["amplitude"] == a]
            e = [r["mean_error"] for r in records
                 if r["method"] == m and r["amplitude"] == a]
            amps.append(a)
            mean.append(np.mean(vals))
            lo.append(np.min(vals))
            hi.append(np.max(vals))
            err.append(np.nanmean(e))
        style = "--o" if m.startswith("thinning") else "-o"
        axes[0].plot(amps, mean, style, color=c, label=m, ms=4)
        axes[0].fill_between(amps, lo, hi, color=c, alpha=0.12)
        axes[1].plot(amps, err, style, color=c, label=m, ms=4)

    axes[0].axhline(EXPECTED_BRANCHES, color="black", ls=":", lw=1.5)
    axes[0].text(0.05, EXPECTED_BRANCHES + 1.5, "true answer: 3 branches",
                 fontsize=9)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_xlabel("boundary noise amplitude (px)")
    axes[0].set_ylabel("branches reported")
    axes[0].set_title("branches found on a shape that has 3")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("boundary noise amplitude (px)")
    axes[1].set_ylabel("mean distance to true centerline (px)")
    # No spin here: aggressive spur pruning wins this panel at high noise,
    # because throwing the spurs away leaves only the well centred core.
    axes[1].set_title("centerline accuracy: comparable,\n"
                      "and not the bisector method at 3px")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    # a picture of the worst case, so the numbers have something to point at
    noisy = phantoms.largest_component(
        phantoms.roughen(clean, amplitude=3.0, scale=3.0, seed=0))
    thin = baseline.thinning(noisy)
    sk = skeletonize(noisy, epsilon=4.0, spacing=1.0, prune=1.0)
    viz.show_mask(axes[2], noisy, cmap="gray")
    viz.draw_pixel_skeleton(axes[2], thin, ms=1.6,
                            label=f"thinning ({baseline.count_pixel_branches(thin)['branches']} branches)")
    viz.draw_skeleton(axes[2], sk, lw=1.8,
                      label=f"bisector eps=4 ({sk.n_branches()} branches)")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].set_title("noise amplitude 3px")

    viz.finish(fig, os.path.join(OUT, "03_noise.png"))

    print("\nbranches reported (mean over seeds)")
    print("amp   " + "".join(f"{m:>18}" for m in methods))
    for a in AMPLITUDES:
        line = f"{a:<6.1f}"
        for m in methods:
            v = [r["branches"] for r in records
                 if r["method"] == m and r["amplitude"] == a]
            line += f"{np.mean(v):>18.1f}"
        print(line)


if __name__ == "__main__":
    main()
