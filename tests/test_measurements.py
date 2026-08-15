"""Tests for the code that produces the numbers, not the geometry.

These exist because a coverage run showed metrics.py and baseline.py at 0%,
which is exactly where both of the bugs that flattered this method lived: the
junction-counting artefact in baseline, the unfair storage comparison in
metrics. The geometry was well covered and correct the whole time; the
measurement code was doing the damage and nothing was watching it.

Every expectation below is worked out by hand, not recorded from a run.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medskel import baseline, phantoms
from medskel.chase import dice
from medskel.metrics import (centerline_error, coverage, skeleton_agreement,
                             vessel_summary, compactness)
from medskel.voronoi import skeletonize


# --- helpers: skeleton images drawn by hand ---------------------------------

def horizontal_line(length=40, shape=(30, 60), row=15, col0=10):
    s = np.zeros(shape, bool)
    s[row, col0:col0 + length] = True
    return s


def plus_sign(shape=(41, 41)):
    """One junction, four arms."""
    s = np.zeros(shape, bool)
    s[20, 5:36] = True
    s[5:36, 20] = True
    return s


# --- baseline: counting -----------------------------------------------------

def test_straight_line_is_one_branch_two_endpoints():
    c = baseline.count_pixel_branches(horizontal_line())
    assert c["branches"] == 1
    assert c["endpoints"] == 2
    assert c["junctions"] == 0


def test_plus_sign_is_four_branches_one_junction():
    """A + has 4 arms meeting at 1 point, so 4 branches and 1 junction.

    On the pixel grid that meeting point is not one pixel. In 8-connectivity
    five mutually touching pixels each end up with 3+ neighbours, so naive
    counting sees five junctions and a fistful of 1px branches. That is the
    artefact, measured here rather than assumed.
    """
    merged = baseline.count_pixel_branches(plus_sign(), merge_junctions=True)
    assert merged["branches"] == 4
    assert merged["endpoints"] == 4
    assert merged["junctions"] == 1

    naive = baseline.count_pixel_branches(plus_sign(), merge_junctions=False)
    assert naive["junctions"] == 5
    assert naive["branches"] > merged["branches"]


def test_junction_knot_is_merged_to_one_junction():
    """The actual artefact: several mutually adjacent junction pixels.

    Three arms meeting diagonally leave a knot of touching pixels each having
    3+ neighbours. Naive counting reports extra 1px branches; merging must
    collapse the knot to a single junction.
    """
    thin = baseline.thinning(phantoms.bifurcation()[0])
    naive = baseline.count_pixel_branches(thin, merge_junctions=False)
    merged = baseline.count_pixel_branches(thin, merge_junctions=True)

    assert merged["branches"] == 3
    assert merged["junctions"] == 1
    # and the artefact is real, not imagined: naive counting inflates it
    assert naive["branches"] > merged["branches"]


def test_polyline_vertices_of_a_straight_line_is_two():
    """A straight line needs exactly its two endpoints, at any tolerance."""
    assert baseline.polyline_vertices(horizontal_line(), simplify=0.5) == 2


def test_a_single_pixel_bump_is_not_counted_as_a_branch():
    """Documents a deliberate consequence of junction merging.

    One pixel sitting on top of a line touches three line pixels, so the whole
    cluster merges into a single pass-through node and the bump disappears. A
    1px bump has no length and is not a branch, so this is the right answer --
    and it errs in the *baseline's* favour, since it removes spurious branches
    from pixel thinning rather than from the bisector method.
    """
    s = horizontal_line(length=40)
    s[14, 25] = True
    assert baseline.count_pixel_branches(s)["branches"] == 1


def test_a_real_short_spur_does_survive_merging():
    """The other side of it: merging must not swallow genuine short branches."""
    s = horizontal_line(length=40)
    for k in range(1, 7):                 # a 6px spur off the trunk
        s[15 - k, 25] = True
    c = baseline.count_pixel_branches(s)
    assert c["branches"] == 3
    assert c["junctions"] == 1

    pruned = baseline.prune_pixel_skeleton(s, min_length=10)
    assert baseline.count_pixel_branches(pruned)["branches"] == 1


# --- metrics: distances -----------------------------------------------------

def test_identical_skeletons_agree_completely():
    s = horizontal_line()
    assert skeleton_agreement(s, s, tol=2.0) == pytest.approx(1.0)


def test_far_apart_skeletons_do_not_agree():
    a = horizontal_line(row=5)
    b = horizontal_line(row=25)          # 20 px away
    assert skeleton_agreement(a, b, tol=2.0) == pytest.approx(0.0)


def test_agreement_is_sensitive_at_the_tolerance():
    """Shifting by 3px should fail a 2px tolerance and pass a 4px one."""
    a = horizontal_line(row=10)
    b = horizontal_line(row=13)
    assert skeleton_agreement(a, b, tol=2.0) == pytest.approx(0.0)
    assert skeleton_agreement(a, b, tol=4.0) == pytest.approx(1.0)


def test_centerline_error_is_zero_against_itself():
    truth = np.column_stack([np.arange(10, 50, 0.5), np.full(80, 15.0)])
    e = centerline_error(horizontal_line(), truth)
    assert e["mean"] < 0.51            # half-pixel rounding only
    assert coverage(horizontal_line(), truth, tol=2.0) == pytest.approx(1.0)


def test_dice_bounds():
    a = np.zeros((20, 20), bool); a[5:15, 5:15] = True
    b = np.zeros((20, 20), bool); b[5:15, 5:15] = True
    c = np.zeros((20, 20), bool); c[0:3, 0:3] = True
    assert dice(a, b) == pytest.approx(1.0)
    assert dice(a, c) == pytest.approx(0.0)
    half = np.zeros((20, 20), bool); half[5:15, 5:10] = True
    # |A|=100, |B|=50, overlap 50  ->  2*50/150
    assert dice(a, half) == pytest.approx(2 / 3)


# --- metrics: vessel_summary ------------------------------------------------

def test_vessel_summary_on_a_tube_of_known_size():
    """A straight tube of radius 18: length and calibre are known in advance."""
    mask, truth = phantoms.straight_tube(radius=18)
    true_length = np.hypot(*(truth[-1] - truth[0]))

    for est in (skeletonize(mask, epsilon=2.0, prune=1.0),
                baseline.thinning(mask)):
        s = vessel_summary(est, mask)
        assert s["total_length"] == pytest.approx(true_length, rel=0.10)
        assert s["median_calibre"] == pytest.approx(36, rel=0.15)
        assert s["median_tortuosity"] == pytest.approx(1.0, abs=0.10)
        assert s["n_bifurcations"] == 0


def test_the_two_representations_measure_a_shape_the_same_way():
    """The inter-observer study compares methods, so systematic disagreement
    between the two measurement paths would land on the result as if it were a
    property of a method."""
    mask, _ = phantoms.bifurcation()
    a = vessel_summary(skeletonize(mask, epsilon=2.0, prune=1.0), mask)
    b = vessel_summary(baseline.thinning(mask), mask)

    assert a["n_bifurcations"] == b["n_bifurcations"]
    assert a["total_length"] == pytest.approx(b["total_length"], rel=0.10)
    assert a["median_calibre"] == pytest.approx(b["median_calibre"], rel=0.15)


def test_compactness_compares_at_matched_tolerance():
    """Regression: it used to compare our simplified polylines against the
    baseline's raw pixel count, which is not a comparison of two skeletons."""
    mask, _ = phantoms.curved_tube()
    thin = baseline.thinning(mask)
    c = compactness(skeletonize(mask, epsilon=2.0, prune=1.0), thin)

    assert c["skeleton_pixels"] == int(np.count_nonzero(thin))
    for tol, row in c["by_tolerance"].items():
        assert row["thinning"] > 0 and row["bisector"] > 0
    # coarser tolerance must not need more points than a finer one
    tols = sorted(c["by_tolerance"], key=float)
    for lo, hi in zip(tols, tols[1:]):
        assert c["by_tolerance"][hi]["thinning"] <= c["by_tolerance"][lo]["thinning"]
        assert c["by_tolerance"][hi]["bisector"] <= c["by_tolerance"][lo]["bisector"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
