"""Figure 2: the paper's wavefront method, where it is exact and where it stops.

Two things get checked here.

1. On convex polygons the wavefront and the Voronoi medial axis agree to
   machine-ish precision, and both agree with the analytic answer. That is the
   cross-validation: two independent implementations of the same definition
   landing in the same place.

2. On anything with a concavity the wavefront stops, because it needs an event
   type that is not implemented. And even if it were implemented, it would give
   a different curve from the medial axis, which the last panel shows.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from medskel import viz
from medskel.bisector import propagate, SplitEventError
from medskel.polygon import Boundary
from medskel.voronoi import skeletonize_polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

SQUARE = np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])
RECT = np.array([[0., 0.], [200., 0.], [200., 40.], [0., 40.]])
PENTAGON = np.column_stack([
    50 + 40 * np.cos(np.linspace(0, 2 * np.pi, 6)[:-1]),
    50 + 40 * np.sin(np.linspace(0, 2 * np.pi, 6)[:-1])])
LSHAPE = np.array([[0., 0.], [100., 0.], [100., 40.],
                   [40., 40.], [40., 100.], [0., 100.]])

ANALYTIC = {
    # 4 half-diagonals to the centre
    "square": 4 * np.hypot(50, 50),
    # central axis (200-40) plus 4 corner diagonals of length 20*sqrt(2)
    "rectangle": 160 + 4 * np.hypot(20, 20),
    # 5 circumradii
    "pentagon": 5 * 40.0,
}


def wavefront_points(result, step=0.5):
    pts = []
    for a, b in result.arcs:
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.arange(n + 1) / n
        pts.append(a + np.outer(t, b - a))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def compare(name, poly):
    """Wavefront vs Voronoi medial axis on one polygon."""
    ma = skeletonize_polygon(Boundary(poly), spacing=0.25, theta_deg=70,
                             prune=0.0)
    try:
        wf = propagate(poly)
        err = np.nan
        if wf.arcs:
            wp = wagon = wavefront_points(wf)
            mp = np.vstack([p for p in ma.polylines()])
            from medskel.metrics import _densify
            mp = np.vstack([_densify(p, 0.5) for p in ma.polylines()])
            d, _ = cKDTree(mp).query(wp)
            err = float(np.percentile(d, 95))
        return wf, ma, err, None
    except SplitEventError as e:
        return None, ma, np.nan, e


def main():
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.0))

    print("convex polygons: wavefront vs medial axis vs analytic")
    for ax, (name, poly) in zip(axes[:3], [("square", SQUARE),
                                           ("rectangle", RECT),
                                           ("pentagon", PENTAGON)]):
        wf, ma, err, exc = compare(name, poly)
        closed = np.vstack([poly, poly[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="black", lw=1.5)

        for i, (a, b) in enumerate(wf.arcs):
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1f77b4", lw=3.0,
                    alpha=0.65, label="wavefront (paper)" if i == 0 else None)
        viz.draw_skeleton(ax, ma, color="#d62728", lw=1.2,
                          label="medial axis (Voronoi)")

        want = ANALYTIC[name]
        got = wf.total_length()
        print(f"  {name:10s} wavefront={got:9.4f}  analytic={want:9.4f}  "
              f"diff={abs(got-want):.2e}  p95 gap to medial axis={err:.3f}px")
        ax.set_title(f"{name}\nlength {got:.2f} vs analytic {want:.2f}\n"
                     f"agreement with medial axis: {err:.2f}px", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="lower center")

    # the concave case
    ax = axes[3]
    wf, ma, err, exc = compare("L", LSHAPE)
    closed = np.vstack([LSHAPE, LSHAPE[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color="black", lw=1.5)
    viz.draw_skeleton(ax, ma, color="#d62728", lw=1.6, label="medial axis")

    partial = propagate(LSHAPE, on_split="stop")
    for i, (a, b) in enumerate(partial.arcs):
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1f77b4", lw=3.0,
                alpha=0.6, label="wavefront, as far as it gets" if i == 0 else None)
    if partial.events:
        t, kind, where = partial.events[-1]
        ax.plot(where[0], where[1], "X", color="black", ms=13, zorder=6)
        ax.annotate(f"{kind}\nt={t:.1f}", xy=where, xytext=(where[0] + 8, where[1] - 22),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))

    ax.annotate("medial axis bends here\n(2.5px off a straight chord):\n"
                "a straight skeleton cannot",
                xy=(24, 24), xytext=(46, 62), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1, color="#d62728"),
                color="#d62728")
    ax.set_title("L-shape: one reflex vertex\nwavefront stops at the "
                 "first event", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")

    print(f"\n  L-shape: {exc}")

    viz.finish(fig, os.path.join(OUT, "02_wavefront.png"))

    curvature_check(ma)


def curvature_check(ma):
    """Is the medial axis of the L actually curved near the reflex vertex?

    It should be: near a reflex vertex the closest boundary feature is a point,
    the other is a line, and the locus equidistant from a point and a line is a
    parabola. A straight skeleton can only ever produce straight segments, so
    this is where the two definitions come apart.
    """
    reflex = np.array([40., 40.])
    best, best_d = None, np.inf
    for poly in ma.polylines():
        d = np.min(np.hypot(*(poly - reflex).T))
        if d < best_d:
            best, best_d = poly, d

    if best is None or len(best) < 5:
        print("\n  (no branch found near the reflex vertex)")
        return

    a, b = best[0], best[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        return
    # perpendicular distance of each point from the chord
    rel = best - a
    dev = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
    print(f"\n  branch nearest the reflex vertex: {len(best)} points, "
          f"chord {L:.1f}px")
    print(f"  max deviation from a straight chord: {dev.max():.2f}px "
          f"({100*dev.max()/L:.1f}% of chord)")
    print("  a straight skeleton arc would have deviation 0 by construction")


if __name__ == "__main__":
    main()
