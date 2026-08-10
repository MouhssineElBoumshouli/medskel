"""How we score a skeleton.

Four things get measured, because "looks right" is not a number:

  centerline error   how far the skeleton sits from the true centerline
  false branches     how many branches exist that should not
  centeredness       does the skeleton stay in the middle of the shape
  compactness        how many numbers it takes to store the result

The first two need a ground truth, so they only apply to phantoms. The last two
work on any mask, which is why they are the ones reported on the real images.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt


def _as_points(obj, step=1.0):
    """Accept a Skeleton, a pixel skeleton image, points, or a list of arms."""
    from .voronoi import Skeleton

    # phantoms with more than one arm hand back a list of polylines
    if isinstance(obj, (list, tuple)):
        parts = [_as_points(o, step) for o in obj]
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else np.zeros((0, 2))

    if isinstance(obj, Skeleton):
        pts = []
        for poly in obj.polylines():
            pts.append(_densify(poly, step))
        return np.vstack(pts) if pts else np.zeros((0, 2))

    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype != bool:
        return arr.astype(float)

    ys, xs = np.nonzero(arr)          # boolean image
    return np.column_stack([xs, ys]).astype(float)


def _densify(poly, step=1.0):
    """Resample a polyline so distance metrics are not biased by vertex spacing."""
    poly = np.asarray(poly, float)
    if len(poly) < 2:
        return poly
    seg = np.diff(poly, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    out = []
    for i, l in enumerate(L):
        n = max(int(np.ceil(l / step)), 1)
        t = np.arange(n) / n
        out.append(poly[i] + np.outer(t, seg[i]))
    out.append(poly[-1:])
    return np.vstack(out)


def centerline_error(estimate, truth, step=1.0, trim=0.0):
    """Distance from the estimated skeleton to the true centerline.

    One directional distance on purpose: every point of the estimate should lie
    on the true centerline, but the true centerline may legitimately not be
    fully covered near the ends, where every skeletonization method stops short
    or fans out. `trim` drops that fraction of the true centerline at each end
    before comparing.
    """
    est = _as_points(estimate, step)
    # trim each arm separately, otherwise on a multi-arm phantom the trim eats
    # the start of one arm and the end of another instead of both tips
    parts = truth if isinstance(truth, (list, tuple)) else [truth]
    gt = []
    for part in parts:
        p = _as_points(part, step)
        if trim > 0 and len(p) > 2:
            k = int(len(p) * trim)
            if len(p) - 2 * k > 1:
                p = p[k:len(p) - k]
        gt.append(p)
    gt = np.vstack([g for g in gt if len(g)]) if gt else np.zeros((0, 2))

    if len(est) == 0 or len(gt) == 0:
        return {"mean": np.nan, "p95": np.nan, "max": np.nan, "n": 0}

    d, _ = cKDTree(gt).query(est)
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)),
            "max": float(d.max()), "n": int(len(est))}


def coverage(estimate, truth, tol=2.0, step=1.0):
    """Fraction of the true centerline that has an estimate point near it.

    Guards against the degenerate way to win at centerline_error, which is to
    return almost nothing and have the little you return be very accurate.
    """
    est = _as_points(estimate, step)
    gt = _as_points(truth, step)
    if len(est) == 0 or len(gt) == 0:
        return 0.0
    d, _ = cKDTree(est).query(gt)
    return float(np.mean(d <= tol))


def centeredness(estimate, mask, step=1.0):
    """Compare the clearance at each skeleton point with the best available.

    For a point on the true medial axis, its distance to the boundary is a
    local maximum of the distance transform. So we look up the distance
    transform at each skeleton point and compare it with the largest value
    within a small neighbourhood. A well centered skeleton scores near 1.
    """
    mask = np.asarray(mask) > 0
    dt = distance_transform_edt(mask)
    est = _as_points(estimate, step)
    if len(est) == 0:
        return {"mean_ratio": np.nan, "mean_clearance": np.nan}

    h, w = mask.shape
    xi = np.clip(np.round(est[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(est[:, 1]).astype(int), 0, h - 1)
    here = dt[yi, xi]

    best = np.zeros_like(here)
    for k, (x, y) in enumerate(zip(xi, yi)):
        y0, y1 = max(y - 2, 0), min(y + 3, h)
        x0, x1 = max(x - 2, 0), min(x + 3, w)
        best[k] = dt[y0:y1, x0:x1].max()

    ok = best > 1e-9
    ratio = np.where(ok, here / np.where(ok, best, 1.0), np.nan)
    return {"mean_ratio": float(np.nanmean(ratio)),
            "mean_clearance": float(here.mean()),
            "outside_fraction": float(np.mean(here <= 0))}


def false_branches(n_found, n_expected):
    """Branches beyond what the phantom actually has."""
    return max(int(n_found) - int(n_expected), 0)


def compactness(skeleton, pixel_skeleton):
    """How many numbers each representation costs.

    A polyline vertex and a skeleton pixel are both 2 coordinates, so this is a
    fair comparison of storage, and it is also a fair comparison of how many
    primitives you have to reason about downstream.
    """
    n_poly = skeleton.n_polyline_vertices()
    n_pix = int(np.count_nonzero(pixel_skeleton))
    return {"polyline_vertices": n_poly, "skeleton_pixels": n_pix,
            "ratio": n_pix / n_poly if n_poly else np.nan}
