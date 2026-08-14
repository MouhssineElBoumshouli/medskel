"""Figure 7: does the noise advantage show up on a real image? No.

experiments/03_noise.py shows that on a synthetic phantom with correlated
noise added to its boundary, this method reports far fewer spurious branches
than pixel thinning. That is a real result about that phantom. It is not, by
itself, a result about medical images, and this script is the check that says
so out loud.

The worry is specific. The retinal vessel mask is built with a morphological
closing, which smooths the boundary before the skeletonizer ever sees it. If
the advantage only exists on rough boundaries, and the segmentation hands over
a smooth one, then the advantage never gets a chance to matter. So: rebuild the
mask at several closing radii, from none to the default, and compare.

The answer is that the two methods agree at every setting, and at eps=2 this
method is sometimes slightly worse. The advantage does not transfer.

Two honest caveats on the negative result, in both directions:

  - There is no ground truth here. On a phantom "3 branches" is a fact. On a
    fundus image nobody knows the true number, so agreeing on ~106 branches
    does not tell us that either method is right, only that they agree.
  - One image, one segmentation pipeline. This is evidence, not proof. It would
    take an annotated dataset (DRIVE, STARE) to make a real claim.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medskel import data as md, viz, baseline
from medskel.voronoi import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

CLOSINGS = [0, 1, 2, 3, 4]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []

    print("rebuilding the vessel mask at several smoothing levels")
    print(f"{'closing':<9}{'mask px':>9}{'thinning':>10}{'ours e=2':>10}"
          f"{'ours e=4':>10}")

    masks = {}
    for cl in CLOSINGS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask, _, _ = md.retina_vessels(closing=cl)
        thin = baseline.thinning(mask)
        row = {
            "closing": cl,
            "mask_px": int(mask.sum()),
            "thinning": baseline.count_pixel_branches(thin)["branches"],
            "ours_eps2": skeletonize(mask, epsilon=2.0, prune=1.0).n_branches(),
            "ours_eps4": skeletonize(mask, epsilon=4.0, prune=1.0).n_branches(),
        }
        rows.append(row)
        masks[cl] = mask
        print(f"{cl:<9}{row['mask_px']:>9}{row['thinning']:>10}"
              f"{row['ours_eps2']:>10}{row['ours_eps4']:>10}")

    with open(os.path.join(RESULTS, "transfer.json"), "w") as f:
        json.dump(rows, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    x = [r["closing"] for r in rows]
    ax.plot(x, [r["thinning"] for r in rows], "--o", color="#ff7f0e",
            label="pixel thinning")
    ax.plot(x, [r["ours_eps2"] for r in rows], "-o", color="#1f77b4",
            label="ours eps=2")
    ax.plot(x, [r["ours_eps4"] for r in rows], "-o", color="#d62728",
            label="ours eps=4")
    ax.set_xlabel("smoothing applied when building the mask (px)")
    ax.set_ylabel("branches reported")
    ax.set_title("on the real image the methods agree\n"
                 "at every smoothing level", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # the roughest and smoothest masks, so "closing" is not just a number
    for ax, cl in zip(axes[1:], [0, 4]):
        viz.show_mask(ax, masks[cl], cmap="gray")
        ax.set_title(f"mask with closing={cl}\n{int(masks[cl].sum())} px",
                     fontsize=11)

    viz.finish(fig, os.path.join(OUT, "07_does_it_transfer.png"))

    print("\nconclusion: the gap measured on the synthetic phantom does not")
    print("appear on this image. Note there is no ground truth here, so this")
    print("says the two methods agree, not that either one is correct.")


if __name__ == "__main__":
    main()
