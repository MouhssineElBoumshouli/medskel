"""Turning a binary segmentation into polygons.

This is the "Cartesianization" step of Saidou et al. (2024): before we can talk
about bisectors we need straight boundary segments, so the pixel mask has to
become a polygon first.

Two knobs, and they do different jobs:

  epsilon  - Ramer-Douglas-Peucker tolerance, in pixels. This is the one that
             actually matters. It decides how much boundary detail survives,
             so it decides how much of the boundary noise gets turned into
             skeleton branches later on.
  spacing  - how finely we re-sample the *simplified* polygon afterwards. This
             is only there for numerical accuracy of the Voronoi step, it does
             not put detail back in.

Keeping them separate is the whole point. With pixel thinning you get one fixed
level of detail (the pixel grid) and you clean up afterwards with pruning
heuristics. Here the smoothing happens before the skeleton exists.
"""

import cv2
import numpy as np


class Boundary:
    """The boundary of one connected component: outer ring + hole rings.

    Rings are (N, 2) float arrays of (x, y), counter-clockwise for the outer
    ring, and not closed (first point is not repeated at the end).
    """

    def __init__(self, outer, holes=None):
        self.outer = np.asarray(outer, dtype=float)
        self.holes = [np.asarray(h, dtype=float) for h in (holes or [])]

    @property
    def rings(self):
        return [self.outer] + self.holes

    def n_vertices(self):
        return sum(len(r) for r in self.rings)


def mask_to_polygon(mask, epsilon=2.0, keep_holes=True, min_area=20.0):
    """Extract the largest component of `mask` and simplify it to a polygon.

    `epsilon` is in pixels and is passed straight to RDP. epsilon=0 keeps every
    boundary pixel, which is the "no regularization" case we compare against.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)

    mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(mask_u8, mode, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no foreground found in the mask")

    # RETR_CCOMP gives a 2-level hierarchy: outer contours first, then holes.
    # hierarchy[0][i] = [next, prev, first_child, parent]
    hierarchy = hierarchy[0] if hierarchy is not None else None

    outer_ids = [i for i in range(len(contours))
                 if hierarchy is None or hierarchy[i][3] < 0]
    biggest = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))

    hole_ids = []
    if keep_holes and hierarchy is not None:
        hole_ids = [i for i in range(len(contours))
                    if hierarchy[i][3] == biggest
                    and cv2.contourArea(contours[i]) >= min_area]

    outer = _simplify(contours[biggest], epsilon)
    holes = [_simplify(contours[i], epsilon) for i in hole_ids]
    holes = [h for h in holes if len(h) >= 3]

    return Boundary(_orient(outer, ccw=True),
                    [_orient(h, ccw=False) for h in holes])


def _simplify(contour, epsilon):
    if epsilon > 0:
        contour = cv2.approxPolyDP(contour, epsilon, True)
    return contour.reshape(-1, 2).astype(float)


def _signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _orient(ring, ccw=True):
    if (_signed_area(ring) > 0) != ccw:
        return ring[::-1].copy()
    return ring


def resample_ring(ring, spacing):
    """Put points every `spacing` pixels along a closed polygon.

    The polygon shape does not change, we only add points along the existing
    edges. Voronoi of a point set only approximates the medial axis of the
    polygon when the boundary samples are dense compared to the local
    thickness, so this matters for accuracy but not for detail.
    """
    closed = np.vstack([ring, ring[:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])

    out = []
    for i, L in enumerate(seg_len):
        if L < 1e-12:
            continue
        n = max(int(np.ceil(L / spacing)), 1)
        t = np.arange(n) / n
        out.append(closed[i] + np.outer(t, seg[i]))
    return np.vstack(out)


def sample_boundary(boundary, spacing=1.0):
    """Dense sample of every ring. Returns (points, ring_id)."""
    pts, ids = [], []
    for k, ring in enumerate(boundary.rings):
        p = resample_ring(ring, spacing)
        pts.append(p)
        ids.append(np.full(len(p), k))
    return np.vstack(pts), np.concatenate(ids)


def polygon_contains(boundary, points):
    """Point-in-polygon for the region (outer ring minus holes).

    cv2.pointPolygonTest wants int32 contours, and rounding a polygon to
    integers is exactly the kind of thing that bites you at the boundary, so
    this uses matplotlib's Path instead which works in float.
    """
    from matplotlib.path import Path

    points = np.atleast_2d(np.asarray(points, dtype=float))
    inside = Path(boundary.outer).contains_points(points)
    for hole in boundary.holes:
        inside &= ~Path(hole).contains_points(points)
    return inside
