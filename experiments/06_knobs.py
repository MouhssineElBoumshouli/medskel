"""Figure 6: which knob causes the fragmentation?

The skeleton of the retina mask comes out in ~29 disconnected pieces even
though the mask is a single connected component. My first guess was that this
was the simplification tolerance: a retinal tree spans an order of magnitude in
width, one global epsilon cannot suit both the arcades and the capillaries, so
epsilon tuned for the wide vessels was cutting the thin ones.

That guess was wrong, and this script is what showed it. The two knobs are
separable: hold one, vary the other. Varying theta moves the component count
from 13 to 39. Varying epsilon barely moves it at all. Epsilon does something
else instead, which is to progressively delete thin vessels.

So they are two different problems and they need two different fixes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medskel import data as md, viz
from medskel.voronoi import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

EPSILONS = [1.0, 2.0, 3.0, 4.0, 6.0]
THETAS = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0]
THIN_RADIUS = 2.0        # "capillary" for the purposes of this figure


def measure(mask, epsilon, theta):
    sk = skeletonize(mask, epsilon=epsilon, spacing=1.0, theta_deg=theta,
                     prune=1.0)
    table = sk.branch_table()
    thin = sum(r["length"] for r in table if r["mean_radius"] < THIN_RADIUS)
    return {
        "epsilon": epsilon, "theta": theta,
        "components": int(sk.meta["n_components_raw"]),
        "branches": sk.n_branches(),
        "total_length": sk.total_length(),
        "thin_length": float(thin),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    mask, _, _ = md.retina_vessels()

    print("varying epsilon at theta=70")
    by_eps = [measure(mask, e, 70.0) for e in EPSILONS]
    for r in by_eps:
        print(f"  eps={r['epsilon']:<4} components={r['components']:<4} "
              f"length={r['total_length']:.0f} "
              f"thin(<{THIN_RADIUS}px)={r['thin_length']:.0f}")

    print("\nvarying theta at epsilon=2")
    by_theta = [measure(mask, 2.0, t) for t in THETAS]
    for r in by_theta:
        print(f"  theta={r['theta']:<5} components={r['components']:<4} "
              f"length={r['total_length']:.0f} "
              f"thin(<{THIN_RADIUS}px)={r['thin_length']:.0f}")

    with open(os.path.join(RESULTS, "knobs.json"), "w") as f:
        json.dump({"by_epsilon": by_eps, "by_theta": by_theta}, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot([r["theta"] for r in by_theta], [r["components"] for r in by_theta],
            "-o", color="#d62728", label="varying theta (eps=2)")
    ax.set_xlabel("separation angle theta (degrees)")
    ax.set_ylabel("disconnected pieces")
    ax.set_title("theta controls fragmentation", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([r["epsilon"] for r in by_eps], [r["components"] for r in by_eps],
            "-o", color="#1f77b4", label="varying eps (theta=70)")
    ax.set_xlabel("simplification tolerance eps (px)")
    ax.set_ylabel("disconnected pieces")
    ax.set_ylim(0, max(r["components"] for r in by_theta) + 5)
    ax.set_title("epsilon does not", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r["epsilon"] for r in by_eps], [r["thin_length"] for r in by_eps],
            "-o", color="#1f77b4")
    ax.set_xlabel("simplification tolerance eps (px)")
    ax.set_ylabel(f"length in vessels thinner than {THIN_RADIUS}px radius")
    ax.set_title("what epsilon does instead:\nit deletes the thin vessels",
                 fontsize=11)
    ax.grid(alpha=0.3)

    viz.finish(fig, os.path.join(OUT, "06_knobs.png"))


if __name__ == "__main__":
    main()
