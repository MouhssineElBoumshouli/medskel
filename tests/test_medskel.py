"""Checks that have a right answer known in advance.

Run with: python -m pytest tests -q   (or just: python tests/test_medskel.py)

Most of these compare against straight skeletons that can be worked out by
hand, which is the only reason they are worth writing: a test that just records
whatever the code did today would not have caught the two indexing bugs that
these did.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medskel import phantoms
from medskel.bisector import propagate, SplitEventError
from medskel.polygon import Boundary, mask_to_polygon, polygon_contains
from medskel.voronoi import skeletonize, skeletonize_polygon

SQUARE = np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]])
RECT = np.array([[0., 0.], [200., 0.], [200., 40.], [0., 40.]])
PENTAGON = np.column_stack([
    50 + 40 * np.cos(np.linspace(0, 2 * np.pi, 6)[:-1]),
    50 + 40 * np.sin(np.linspace(0, 2 * np.pi, 6)[:-1])])
LSHAPE = np.array([[0., 0.], [100., 0.], [100., 40.],
                   [40., 40.], [40., 100.], [0., 100.]])


# --- the paper's wavefront, against lengths worked out by hand --------------

@pytest.mark.parametrize("poly,expected", [
    (SQUARE, 4 * np.hypot(50, 50)),          # 4 half-diagonals
    (RECT, 160 + 4 * np.hypot(20, 20)),      # central axis + 4 corner arcs
    (PENTAGON, 5 * 40.0),                    # 5 circumradii
])
def test_wavefront_matches_analytic_length(poly, expected):
    result = propagate(poly)
    assert not result.stopped_early
    assert result.total_length() == pytest.approx(expected, abs=1e-6)


def test_rectangle_keeps_its_central_axis():
    """Regression: the final degenerate wavefront is skeleton too.

    Dropping it cost the rectangle all 160px of its central axis and left only
    the four corner arcs.
    """
    result = propagate(RECT)
    xs = np.concatenate([[a[0], b[0]] for a, b in result.arcs])
    ys = np.concatenate([[a[1], b[1]] for a, b in result.arcs])
    assert xs.min() == pytest.approx(0.0, abs=1e-6)
    assert xs.max() == pytest.approx(200.0, abs=1e-6)
    assert 20.0 == pytest.approx(np.median(ys), abs=1e-6)


def test_wavefront_stays_inside_the_polygon():
    """Regression: a wraparound merge used to emit a node outside the shape.

    "Inside or on the boundary", not "strictly inside": every arc starts at a
    polygon vertex, and those sit exactly on the boundary.
    """
    from scipy.spatial import cKDTree
    from medskel.polygon import resample_ring

    result = propagate(RECT)
    pts = np.array([p for arc in result.arcs for p in arc])
    outside = ~polygon_contains(Boundary(RECT), pts)
    if outside.any():
        d, _ = cKDTree(resample_ring(RECT, 0.5)).query(pts[outside])
        assert d.max() < 1e-6, f"a point escaped by {d.max():.3f}px"


def test_wavefront_refuses_the_concave_case():
    with pytest.raises(SplitEventError):
        propagate(LSHAPE)


def test_wavefront_can_report_instead_of_raising():
    result = propagate(LSHAPE, on_split="stop")
    assert result.stopped_early
    assert result.events


# --- the Voronoi medial axis ------------------------------------------------

@pytest.mark.parametrize("poly,expected", [
    (SQUARE, 4 * np.hypot(50, 50)),
    (PENTAGON, 5 * 40.0),
])
def test_medial_axis_agrees_with_wavefront(poly, expected):
    """Two independent constructions of the same definition should agree."""
    ma = skeletonize_polygon(Boundary(poly), spacing=0.25, prune=0.0)
    assert ma.total_length() == pytest.approx(expected, rel=0.02)


def test_rectangle_topology():
    """A rectangle's skeleton is 5 branches meeting at 2 junctions."""
    mask = np.zeros((120, 300), bool)
    mask[40:80, 50:250] = True
    sk = skeletonize(mask, epsilon=1.0, prune=0.0)
    assert sk.n_branches() == 5
    assert sk.n_bifurcations() == 2


def test_bifurcation_phantom_has_one_bifurcation():
    mask, _ = phantoms.bifurcation()
    sk = skeletonize(mask, epsilon=2.0, prune=1.0)
    assert sk.n_branches() == 3
    assert sk.n_bifurcations() == 1
    assert sk.n_endpoints() == 3


def test_annulus_skeleton_is_a_loop_with_no_ends():
    """A ring has a hole, so its medial axis closes on itself."""
    h = w = 220
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    mask = (r < 90) & (r > 55)

    boundary = mask_to_polygon(mask, epsilon=1.0)
    assert len(boundary.holes) == 1

    sk = skeletonize(mask, epsilon=1.0, prune=1.0)
    assert sk.n_endpoints() == 0
    # radius along the loop is half the ring width
    assert np.median(sk.radii()) == pytest.approx(17.5, abs=2.0)


def test_skeleton_stays_inside_the_mask():
    mask, _ = phantoms.curved_tube()
    sk = skeletonize(mask, epsilon=2.0, prune=1.0)
    pts = np.vstack(sk.polylines())
    inside = mask[np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1),
                  np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)]
    assert inside.mean() > 0.99


def test_simplification_shrinks_storage_but_not_reported_length():
    """Lengths are measured before RDP, so compression must not shorten them.

    The earlier version of this test compared a skeleton against a no-op copy
    of itself, so it asserted that a number equalled itself and could never
    have failed. This one actually varies the tolerance.
    """
    from medskel.voronoi import _reduce_to_branches

    mask, _ = phantoms.curved_tube()
    sk = skeletonize(mask, epsilon=2.0, prune=1.0)

    lengths, sizes = [], []
    for tol in (0.0, 0.25, 1.0):
        sk.branches = _reduce_to_branches(sk.graph, simplify=tol)
        lengths.append(sk.total_length())
        sizes.append(sk.n_polyline_vertices())

    # every tolerance reports the same total length...
    assert lengths[1] == pytest.approx(lengths[0])
    assert lengths[2] == pytest.approx(lengths[0])
    # ...while actually storing fewer points
    assert sizes[0] > sizes[1] > sizes[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
