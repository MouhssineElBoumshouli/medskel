"""Figure 1: the whole pipeline on one retinal image.

fundus -> vessel mask -> polygon -> bisectors -> pruned skeleton graph
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from medskel import data as md, viz
from medskel.polygon import mask_to_polygon, sample_boundary, polygon_contains
from medskel.voronoi import skeletonize

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")

EPSILON = 2.0
ZOOM_HALF = 62          # half-width of the close-up window, in pixels


def raw_voronoi_edges(boundary, spacing=1.0):
    """The unfiltered bisector network, to show what the angle test removes."""
    samples, _ = sample_boundary(boundary, spacing=spacing)
    vor = Voronoi(samples)
    inside = polygon_contains(boundary, vor.vertices)
    segs = []
    for v1, v2 in vor.ridge_vertices:
        if v1 >= 0 and v2 >= 0 and inside[v1] and inside[v2]:
            segs.append((vor.vertices[v1], vor.vertices[v2]))
    return segs


def pick_bifurcation(skel):
    """A bifurcation with three decently long branches, for the close-up."""
    b = skel.branches
    best, best_score = None, -np.inf
    for n, deg in b.degree():
        if deg < 3:
            continue
        lengths = sorted((d["length"] for *_, d in b.edges(n, data=True)),
                         reverse=True)
        score = min(lengths[:3]) + 0.3 * b.nodes[n]["radius"]
        if score > best_score:
            best, best_score = n, score
    if best is None:
        pts = skel.points()
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    p = b.nodes[best]["pos"]
    return float(p[0]), float(p[1])


def main():
    print("segmenting retina...")
    mask, green, vesselness = md.retina_vessels()
    boundary = mask_to_polygon(mask, epsilon=EPSILON)
    print(f"  mask {mask.sum()} px, polygon {boundary.n_vertices()} vertices")

    print("computing skeleton...")
    skel = skeletonize(mask, epsilon=EPSILON, spacing=1.0, prune=1.0)
    print(" ", skel)

    # Same construction, but on the unsimplified pixel boundary. This is what
    # the polygon step is for: on a staircase boundary every little step
    # generates its own bisector, and you get a hair for each one.
    pixel_boundary = mask_to_polygon(mask, epsilon=0.0)
    raw = raw_voronoi_edges(pixel_boundary)
    print(f"  bisectors from raw pixel boundary "
          f"({pixel_boundary.n_vertices()} vertices): {len(raw)} edges")

    cx, cy = pick_bifurcation(skel)
    x0, x1 = cx - ZOOM_HALF, cx + ZOOM_HALF
    y0, y1 = cy - ZOOM_HALF, cy + ZOOM_HALF
    print(f"  close-up centred on bifurcation at ({cx:.0f}, {cy:.0f})")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))

    axes[0].imshow(green, cmap="gray")
    axes[0].set_title("1. fundus photograph\n(green channel)", fontsize=11)

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"2. vessel mask\nFrangi + hysteresis, {mask.sum()} px",
                      fontsize=11)
    axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="#d62728", lw=1.5))

    # from here on we are zoomed in, otherwise nothing is visible
    axes[2].imshow(mask, cmap="gray")
    viz.draw_polygon(axes[2], boundary, markers=True, ms=3.5, lw=1.0)
    axes[2].set_title(f"3. polygon, eps={EPSILON}px\n"
                      f"{boundary.n_vertices()} vertices", fontsize=11)

    axes[3].imshow(mask, cmap="gray", alpha=0.30)
    for a, b in raw:
        axes[3].plot([a[0], b[0]], [a[1], b[1]], "-", color="#9467bd", lw=0.55)
    axes[3].set_title(f"4. same thing without step 3\n"
                      f"bisectors of the raw pixel boundary", fontsize=11)

    axes[4].imshow(mask, cmap="gray", alpha=0.30)
    viz.draw_skeleton(axes[4], skel, nodes=True, lw=1.8)
    axes[4].set_title(f"5. after the angle test + pruning\n"
                      f"{skel.n_branches()} branches, "
                      f"{skel.n_bifurcations()} bifurcations", fontsize=11)

    for ax in axes[2:]:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    viz.finish(fig, os.path.join(OUT, "01_pipeline.png"))


if __name__ == "__main__":
    main()
