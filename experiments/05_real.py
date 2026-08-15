"""Figure 5: two real images, and the measurements you actually get out.

The phantoms say the method is correct. This says it is useful, which is a
different claim. Both cases are here because they stress different things:

  retina    a branching tree. What you want from it is a table of vessels:
            how long, how wide, how tortuous, meeting at how many
            bifurcations. That table is a graph query, not an image operation.
  skull     an annulus. Its medial axis is a closed loop with no free ends at
            all, and the clearance radius along that loop is half the local
            bone thickness, so the skeleton is a thickness measurement.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from medskel import data as md, viz, baseline
from medskel.voronoi import skeletonize
from medskel.metrics import compactness, centeredness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")


def colour_by_radius(ax, skel, cmap="viridis", lw=2.0):
    """Draw the skeleton with each branch coloured by its mean radius."""
    radii = [np.mean(d["radii"]) for *_, d in skel.branches.edges(data=True)]
    vmin, vmax = np.percentile(radii, [2, 98])
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    for (*_, d) in skel.branches.edges(data=True):
        p = d["polyline"]
        ax.plot(p[:, 0], p[:, 1], "-", lw=lw,
                color=cm(norm(np.mean(d["radii"]))), solid_capstyle="round")
    return plt.cm.ScalarMappable(norm=norm, cmap=cm)


def longest_cycle(skel):
    """The vault loop: positions and radii in order around the biggest cycle."""
    cycles = nx.cycle_basis(skel.graph)
    if not cycles:
        return None, None
    nodes = max(cycles, key=len)
    pos = np.array([skel.graph.nodes[n]["pos"] for n in nodes])
    rad = np.array([skel.graph.nodes[n]["radius"] for n in nodes])
    return pos, rad


def main():
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}

    # ---------------- retina ----------------
    print("retina...")
    mask, green, _ = md.retina_vessels()

    t0 = time.time()
    skel = skeletonize(mask, epsilon=2.0, spacing=1.0, prune=1.0)
    t_ours = time.time() - t0

    t0 = time.time()
    thin = baseline.thinning(mask)
    t_thin = time.time() - t0
    thin_counts = baseline.count_pixel_branches(thin)

    comp = compactness(skel, thin)
    table = skel.branch_table()
    lengths = np.array([r["length"] for r in table])
    tort = np.array([r["tortuosity"] for r in table])
    calibre = np.array([2 * r["mean_radius"] for r in table])

    summary["retina"] = {
        "mask_px": int(mask.sum()),
        "polygon_vertices": int(skel.meta["n_polygon_vertices"]),
        "bisector_branches": skel.n_branches(),
        "bisector_bifurcations": skel.n_bifurcations(),
        "bisector_graph_components": int(skel.meta["n_components_raw"]),
        "bisector_seconds": t_ours,
        "thinning_branches": thin_counts["branches"],
        "thinning_junctions": thin_counts["junctions"],
        "thinning_seconds": t_thin,
        "polyline_vertices": comp["polyline_vertices"],
        "skeleton_pixels": comp["skeleton_pixels"],
        "storage_by_tolerance": comp["by_tolerance"],
        "centeredness": centeredness(skel, mask)["mean_ratio"],
        "total_length_px": skel.total_length(),
        "median_calibre_px": float(np.median(calibre)),
        "median_tortuosity": float(np.nanmedian(tort)),
    }
    for k, v in summary["retina"].items():
        print(f"   {k}: {v}")

    # ---------------- skull ----------------
    print("\nskull...")
    bone, slice_img = md.skull_vault()
    skull = skeletonize(bone, epsilon=2.0, spacing=1.0, prune=1.0)
    loop_pos, loop_rad = longest_cycle(skull)
    summary["skull"] = {
        "mask_px": int(bone.sum()),
        "branches": skull.n_branches(),
        "endpoints": skull.n_endpoints(),
        "holes": len(skull.boundary.holes),
        "loop_points": 0 if loop_pos is None else len(loop_pos),
        "median_thickness_px": None if loop_rad is None
        else float(2 * np.median(loop_rad)),
    }
    for k, v in summary["skull"].items():
        print(f"   {k}: {v}")

    with open(os.path.join(RESULTS, "real_images.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    viz.show_mask(ax, mask, cmap="gray")
    viz.draw_pixel_skeleton(ax, thin, ms=0.8)
    ax.set_title(f"retina, pixel thinning\n{thin_counts['branches']} branches, "
                 f"{thin_counts['n_pixels']} px", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mask, cmap="gray", alpha=0.25)
    sm = colour_by_radius(ax, skel, lw=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046)
    cb.set_label("vessel radius (px)", fontsize=9)
    ax.set_title(f"bisector method, coloured by calibre\n{skel.n_branches()} branches, "
                 f"{skel.n_bifurcations()} bifurcations", fontsize=11)

    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(lengths, tort, c=calibre, cmap="viridis", s=18,
                    edgecolor="none")
    ax.set_xscale("log")
    ax.set_ylim(0.98, max(1.35, np.nanpercentile(tort, 98)))
    ax.set_xlabel("branch length (px)")
    ax.set_ylabel("tortuosity (arc / chord)")
    ax.set_title("what the graph gives you:\none row per vessel segment",
                 fontsize=11)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.046).set_label("calibre (px)", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(slice_img, cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    # skimage ships this as data.brain(), from the UNC volume rendering test
    # set. That set contains both a CT and an MR head at 256x256, and the
    # docstring does not say which this is, so it is left unnamed here. What
    # matters for this figure is only that cortical bone is the brightest
    # thing in it.
    ax.set_title("head scan, skimage.data.brain()\n"
                 "(bone is the brightest structure)", fontsize=11)

    ax = fig.add_subplot(gs[1, 1])
    viz.show_mask(ax, bone, cmap="gray")
    viz.draw_skeleton(ax, skull, nodes=True, lw=1.5)
    ax.set_title(f"bone mask + medial axis\n{skull.n_branches()} branches, "
                 f"{len(skull.boundary.holes)} hole", fontsize=11)

    ax = fig.add_subplot(gs[1, 2])
    if loop_pos is not None:
        order = _order_loop(loop_pos)
        p, r = loop_pos[order], loop_rad[order]
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        ax.plot(s, 2 * r, "-", color="#d62728", lw=1.2)
        ax.set_xlabel("distance around the vault (px)")
        ax.set_ylabel("bone thickness (px)")
        ax.set_title(f"thickness along the closed loop\n"
                     f"median {2*np.median(r):.1f} px", fontsize=11)
        ax.grid(alpha=0.3)

    viz.finish(fig, os.path.join(OUT, "05_real.png"))


def _order_loop(pos):
    """Nearest-neighbour walk, cycle_basis does not return nodes in order."""
    n = len(pos)
    remaining = set(range(1, n))
    order = [0]
    while remaining:
        last = pos[order[-1]]
        nxt = min(remaining, key=lambda i: np.hypot(*(pos[i] - last)))
        order.append(nxt)
        remaining.discard(nxt)
    return np.array(order)


if __name__ == "__main__":
    main()
