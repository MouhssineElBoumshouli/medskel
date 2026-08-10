"""Plot helpers. Nothing clever, just keeps the figure scripts short."""

import matplotlib.pyplot as plt
import numpy as np

MASK_CMAP = "gray"
SKEL_COLOR = "#d62728"
POLY_COLOR = "#1f77b4"
BASE_COLOR = "#ff7f0e"
TRUTH_COLOR = "#2ca02c"


def show_mask(ax, mask, alpha=1.0, cmap=MASK_CMAP):
    ax.imshow(np.asarray(mask), cmap=cmap, alpha=alpha, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def draw_polygon(ax, boundary, color=POLY_COLOR, lw=1.2, markers=False, ms=3):
    for ring in boundary.rings:
        closed = np.vstack([ring, ring[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color=color, lw=lw)
        if markers:
            ax.plot(ring[:, 0], ring[:, 1], "o", color=color, ms=ms,
                    markerfacecolor="white", markeredgewidth=0.8)


def draw_skeleton(ax, skeleton, color=SKEL_COLOR, lw=1.6, nodes=False,
                  label=None):
    first = True
    for poly in skeleton.polylines():
        ax.plot(poly[:, 0], poly[:, 1], "-", color=color, lw=lw,
                label=label if first else None, solid_capstyle="round")
        first = False
    if nodes:
        b = skeleton.branches
        for n, d in b.degree():
            p = b.nodes[n]["pos"]
            if d >= 3:
                ax.plot(p[0], p[1], "o", color="black", ms=4.5, zorder=5)
            elif d == 1:
                ax.plot(p[0], p[1], "s", color=color, ms=3.5, zorder=5)


def draw_pixel_skeleton(ax, skel, color=BASE_COLOR, ms=0.7, label=None):
    ys, xs = np.nonzero(np.asarray(skel))
    ax.plot(xs, ys, ".", color=color, ms=ms, label=label)


def draw_centerline(ax, centerline, color=TRUTH_COLOR, lw=2.0, ls="--",
                    label=None):
    parts = centerline if isinstance(centerline, list) else [centerline]
    for i, c in enumerate(parts):
        c = np.asarray(c)
        ax.plot(c[:, 0], c[:, 1], ls, color=color, lw=lw,
                label=label if i == 0 else None)


def panel_grid(n, ncols=3, size=4.0, aspect=1.0):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(size * ncols, size * aspect * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def finish(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
