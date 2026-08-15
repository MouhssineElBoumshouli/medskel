"""Figure 8: does the method survive real expert disagreement?

The noise experiment (03) perturbs a phantom boundary with noise I wrote
myself, and the method wins. The transfer experiment (07) finds no advantage on
one real image. The obvious complaint about both is that neither uses a real,
independently measured boundary perturbation. This one does.

CHASE_DB1 gives 28 retinal images each segmented by *two* people. Two experts
tracing the same vessel disagree about where its edge is, and that disagreement
is exactly the thing medskel claims to absorb. So:

    for each image, skeletonize observer 1's mask and observer 2's mask,
    and ask how much the resulting measurements moved.

A method robust to boundary variation should move less. Both methods see the
identical pair of masks, so any difference between them is attributable to how
they handle the boundary and not to the data.

Everything is a *relative* difference, |a-b| / mean(a,b), because a method that
simply reports fewer branches would show smaller absolute differences without
being any more reliable.

Read the Dice column in the printout first. If the two observers overlap
poorly, they are disagreeing about which vessels exist rather than about where
the edges are, and this experiment is measuring the wrong thing.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from medskel import chase, viz, baseline
from medskel.voronoi import skeletonize
from medskel.metrics import vessel_summary, skeleton_agreement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

MEASURES = ["total_length", "n_bifurcations", "median_calibre",
            "median_tortuosity"]
METHODS = ["thinning", "ours eps=2", "ours eps=4"]
LIMIT = int(os.environ.get("CHASE_LIMIT", "0"))    # 0 = all cases


def skeleton_for(method, mask):
    if method == "thinning":
        return baseline.thinning(mask)
    eps = float(method.split("=")[1])
    return skeletonize(mask, epsilon=eps, prune=1.0)


def relative_difference(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    mean = 0.5 * (abs(a) + abs(b))
    return abs(a - b) / mean if mean > 1e-12 else np.nan


def main():
    os.makedirs(RESULTS, exist_ok=True)

    try:
        cases = chase.load_cases()
    except chase.ChaseNotAvailable as e:
        print(e)
        return 1

    if LIMIT:
        cases = cases[:LIMIT]
    print(f"loaded {len(cases)} cases with two observers each\n")

    records = []
    print(f"{'case':<7}{'dice':>7}" +
          "".join(f"{m:>14}" for m in METHODS) + "   (skeleton agreement)")

    for c in cases:
        d = chase.dice(c["obs1"], c["obs2"])
        row = {"case": c["name"], "dice": d}

        line = f"{c['name']:<7}{d:>7.3f}"
        for method in METHODS:
            s1 = skeleton_for(method, c["obs1"])
            s2 = skeleton_for(method, c["obs2"])

            m1 = vessel_summary(s1, c["obs1"])
            m2 = vessel_summary(s2, c["obs2"])
            for k in MEASURES:
                row[f"{method}|{k}"] = relative_difference(m1[k], m2[k])

            agree = skeleton_agreement(s1, s2, tol=2.0)
            row[f"{method}|agreement"] = agree
            line += f"{agree:>14.3f}"

        print(line)
        records.append(row)

    with open(os.path.join(RESULTS, "interobserver.json"), "w") as f:
        json.dump(records, f, indent=1)

    report(records)
    figure(records, cases)
    return 0


def report(records):
    dices = [r["dice"] for r in records]
    print(f"\ninter-observer Dice: median {np.median(dices):.3f}, "
          f"range {min(dices):.3f}-{max(dices):.3f}")
    if np.median(dices) < 0.75:
        print("  WARNING: low overlap. The observers may disagree about which")
        print("  vessels exist, not just where their edges are, which would")
        print("  confound everything below.")

    print("\nrelative change in each measurement between the two observers")
    print("(lower is better: the measurement moved less when the tracer "
          "changed)\n")
    print(f"{'measure':<20}" + "".join(f"{m:>14}" for m in METHODS) +
          f"{'ours4 vs thin':>16}")

    summary = {}
    for k in MEASURES + ["agreement"]:
        line = f"{k:<20}"
        vals = {}
        for m in METHODS:
            v = np.array([r[f"{m}|{k}"] for r in records], float)
            vals[m] = v
            line += f"{np.nanmedian(v):>14.3f}"

        a = vals["thinning"]
        b = vals["ours eps=4"]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 6 and not np.allclose(a[ok], b[ok]):
            stat, p = wilcoxon(a[ok], b[ok])
            line += f"{'p=' + format(p, '.4f'):>16}"
        else:
            p = np.nan
            line += f"{'n/a':>16}"
        summary[k] = {m: float(np.nanmedian(vals[m])) for m in METHODS}
        summary[k]["wilcoxon_p_ours4_vs_thinning"] = None if not np.isfinite(p) else float(p)
        print(line)

    print("\nagreement is a fraction where higher is better; the other rows")
    print("are relative differences where lower is better.")

    with open(os.path.join(RESULTS, "interobserver_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def figure(records, cases):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    c = cases[0]
    overlap = np.zeros(c["obs1"].shape + (3,), float)
    overlap[..., 0] = c["obs1"]
    overlap[..., 1] = c["obs2"]
    overlap[..., 2] = c["obs1"] & c["obs2"]
    ax.imshow(1 - overlap)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{c['name']}: where the two experts differ\n"
                 f"Dice {chase.dice(c['obs1'], c['obs2']):.3f}", fontsize=11)

    ax = axes[1]
    data = [[r[f"{m}|total_length"] for r in records] for m in METHODS]
    ax.boxplot(data, tick_labels=[m.replace("ours ", "") for m in METHODS])
    ax.set_ylabel("relative change in total vessel length")
    ax.set_title("how much the measurement moves\nwhen the tracer changes",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    data = [[r[f"{m}|agreement"] for r in records] for m in METHODS]
    ax.boxplot(data, tick_labels=[m.replace("ours ", "") for m in METHODS])
    ax.set_ylabel("fraction of skeleton within 2px of the other observer's")
    ax.set_title("do the two skeletons trace\nthe same curves?", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    viz.finish(fig, os.path.join(OUT, "08_interobserver.png"))


if __name__ == "__main__":
    sys.exit(main())
