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


def vessel_summary(estimate, mask):
    """The four numbers a vessel morphometry study would report.

    Accepts either a Skeleton or a boolean pixel skeleton, and computes the
    same quantities the same way for both, because the inter-observer
    experiment compares methods and any difference in the measurement code
    would land on the result as if it were a difference in the method.

    One systematic caveat: arc length along a pixel skeleton is inflated by
    the staircase, roughly 8% on diagonal runs. That biases the pixel method's
    absolute lengths, but the experiment compares observer A against observer B
    within a method, so the bias cancels.
    """
    from .voronoi import Skeleton
    from .baseline import branch_paths

    if isinstance(estimate, Skeleton):
        table = estimate.branch_table()
        if not table:
            return _empty_summary()
        return {
            "total_length": estimate.total_length(),
            "n_bifurcations": estimate.n_bifurcations(),
            "median_calibre": float(np.median([2 * r["mean_radius"]
                                               for r in table])),
            "median_tortuosity": float(np.nanmedian([r["tortuosity"]
                                                     for r in table])),
        }

    paths, g = branch_paths(estimate)
    if not paths:
        return _empty_summary()

    dt = distance_transform_edt(np.asarray(mask) > 0)
    h, w = dt.shape

    lengths, calibres, torts = [], [], []
    for p in paths:
        seg = np.diff(p, axis=0)
        L = float(np.hypot(seg[:, 0], seg[:, 1]).sum()) if len(p) > 1 else 0.0
        chord = float(np.linalg.norm(p[-1] - p[0]))
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        lengths.append(L)
        calibres.append(2 * float(np.mean(dt[yi, xi])))
        if chord > 1e-9 and L > 0:
            torts.append(L / chord)

    return {
        "total_length": float(np.sum(lengths)),
        "n_bifurcations": sum(1 for _, d in g.degree() if d >= 3),
        "median_calibre": float(np.median(calibres)),
        "median_tortuosity": float(np.median(torts)) if torts else np.nan,
    }


def _empty_summary():
    return {"total_length": 0.0, "n_bifurcations": 0,
            "median_calibre": np.nan, "median_tortuosity": np.nan}


def skeleton_agreement(a, b, mask_shape=None, tol=2.0, step=1.0):
    """Fraction of each skeleton lying within `tol` px of the other.

    A direct measure of whether the two observers' skeletons trace the same
    curves, independent of any derived measurement.
    """
    pa, pb = _as_points(a, step), _as_points(b, step)
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float(0.5 * (np.mean(da <= tol) + np.mean(db <= tol)))


def compactness(skeleton, pixel_skeleton, tolerances=(0.25, 0.5, 1.0, 2.0)):
    """Storage cost of each representation, at matched simplification.

    An earlier version of this compared our *simplified* polylines against the
    baseline's *raw pixel count* and reported ratios of 10x to 134x. That was
    measuring our simplification step against the baseline's lack of one, not
    measuring the two skeletons, and it made the method look far better than it
    is.

    Done properly, both skeletons are turned into polylines and simplified at
    the same tolerance. The advantage then turns out to be almost entirely a
    function of that tolerance: large at 0.25px, gone by 1px. Which makes
    sense, because a pixel skeleton is a staircase pinned to the grid, so
    demanding sub-pixel fidelity of it forces you to store every step, and it
    was never accurate to better than about half a pixel in the first place.

    The tolerance sweep is returned rather than a single headline number, so
    the dependence is visible instead of hidden behind a choice.
    """
    from .baseline import polyline_vertices
    from .voronoi import _reduce_to_branches

    n_pix = int(np.count_nonzero(pixel_skeleton))
    sweep = {}
    for tol in tolerances:
        theirs = polyline_vertices(pixel_skeleton, simplify=tol)
        ours = int(sum(len(d["polyline"]) for *_, d in
                       _reduce_to_branches(skeleton.graph, simplify=tol
                                           ).edges(data=True)))
        sweep[str(tol)] = {"thinning": theirs, "bisector": ours,
                           "ratio": theirs / ours if ours else np.nan}

    return {"skeleton_pixels": n_pix,
            "polyline_vertices": skeleton.n_polyline_vertices(),
            "by_tolerance": sweep}
