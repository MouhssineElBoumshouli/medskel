"""The paper's method, done properly: bisectors as a propagating wavefront.

Saidou et al. describe the bisectors as "beams of fire moving to the interior of
the polygon", and say to intersect the bisectors of *neighbouring* vertices and
take the one that meets first (their "prioritized bisector"). So this is not a
one-shot construction. It is a simulation: shrink the polygon inward at
constant speed, and every time an edge collapses to nothing, that is a skeleton
node. Then keep going with the smaller polygon.

That object has a name too. It is the straight skeleton (Aichholzer et al.
1995), and it is not the same thing as the medial axis, which matters here and
is discussed at the bottom of this file.

What is implemented and what is not
-----------------------------------
Two kinds of event can happen while the wavefront shrinks:

  edge event   an edge shrinks to zero length and its two vertices merge.
               Implemented.
  split event  a reflex vertex (interior angle > 180 degrees) runs into an
               edge on the far side and cuts the polygon into two pieces.
               Detected, not implemented.

Split events are the hard half, and every shape with a concavity has them.
Rather than let the simulation quietly produce a wrong answer when one comes
up, propagate() stops and reports it. The point of this module is to be able to
show precisely where the wavefront formulation runs out, so it should fail
loudly.
"""

import numpy as np

from .polygon import polygon_contains, Boundary


class SplitEventError(RuntimeError):
    """Raised when the wavefront needs a split event we do not handle."""

    def __init__(self, message, time, vertex, position):
        super().__init__(message)
        self.time = time
        self.vertex = vertex
        self.position = position


class WavefrontResult:
    def __init__(self, arcs, nodes, events, stopped_early, reason=None):
        self.arcs = arcs                # list of (p, q) skeleton segments
        self.nodes = np.array(nodes) if len(nodes) else np.zeros((0, 2))
        self.events = events            # (time, kind, position)
        self.stopped_early = stopped_early
        self.reason = reason

    def total_length(self):
        return float(sum(np.linalg.norm(q - p) for p, q in self.arcs))

    def __repr__(self):
        state = "incomplete" if self.stopped_early else "complete"
        return (f"<Wavefront {len(self.arcs)} arcs, {len(self.events)} events, "
                f"{state}>")


def _inward_normals(poly):
    """Unit normal of each edge, pointing into the polygon."""
    n = len(poly)
    d = np.roll(poly, -1, axis=0) - poly
    L = np.hypot(d[:, 0], d[:, 1])
    d = d / L[:, None]
    normals = np.column_stack([-d[:, 1], d[:, 0]])

    # decide the sign once, from the polygon's orientation, then check it
    mid = poly + 0.5 * (np.roll(poly, -1, axis=0) - poly)
    probe = mid + 1e-3 * L[:, None] * normals
    inside = polygon_contains(Boundary(poly), probe)
    if inside.mean() < 0.5:
        normals = -normals
    return normals


def _vertex_velocities(poly, normals):
    """Speed and direction of each vertex of the shrinking polygon.

    Vertex i sits on the offset lines of edges i-1 and i. Both lines move
    inward at unit speed, so the vertex velocity u must satisfy
    n_{i-1}.u = 1 and n_i.u = 1. Solving that 2x2 system is exactly the angle
    bisector of the paper, with the 1/sin(angle/2) speed-up that a sharp corner
    needs to keep up with both of its edges.
    """
    n = len(poly)
    u = np.zeros((n, 2))
    reflex = np.zeros(n, bool)
    sign = _orientation_sign(poly)

    for i in range(n):
        a = normals[(i - 1) % n]
        b = normals[i]
        A = np.array([a, b])
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            u[i] = b                       # straight vertex, edges parallel
        else:
            u[i] = np.linalg.solve(A, np.ones(2))

        # reflex test: cross product of the two edge directions
        prev_dir = poly[i] - poly[(i - 1) % n]
        next_dir = poly[(i + 1) % n] - poly[i]
        cross = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
        reflex[i] = cross * sign < 0

    return u, reflex


def _orientation_sign(poly):
    x, y = poly[:, 0], poly[:, 1]
    return np.sign(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _edge_event_times(pos, vel):
    """When each edge collapses to zero length."""
    n = len(pos)
    dp = np.roll(pos, -1, axis=0) - pos
    du = np.roll(vel, -1, axis=0) - vel
    denom = np.einsum("ij,ij->i", du, du)

    t = np.full(n, np.inf)
    ok = denom > 1e-14
    t[ok] = -np.einsum("ij,ij->i", dp[ok], du[ok]) / denom[ok]
    t[~np.isfinite(t)] = np.inf
    t[t <= 1e-9] = np.inf

    # only count it if the edge really does vanish, not just get shorter
    finite = np.isfinite(t)
    if finite.any():
        residual = dp[finite] + t[finite, None] * du[finite]
        bad = np.hypot(residual[:, 0], residual[:, 1]) > 1e-6
        idx = np.flatnonzero(finite)[bad]
        t[idx] = np.inf
    return t


def _collapse_degenerate(pos, edge_normals, tol=1e-7):
    """Drop edges that already have zero length.

    Needed because events are often simultaneous. A rectangle collapses its
    left and right edges at exactly the same instant, but the loop only handles
    one event per step, so the other one is left behind as a zero-length edge.
    Its collapse time then computes as t=0, gets filtered out as "already
    happened", and the wavefront limps on with a phantom vertex in it. That was
    good for a skeleton node at (180, -140) on a polygon that stops at y=0.
    """
    while len(pos) >= 3:
        d = np.roll(pos, -1, axis=0) - pos
        lengths = np.hypot(d[:, 0], d[:, 1])
        k = int(np.argmin(lengths))
        if lengths[k] > tol:
            break
        pos = np.roll(pos, -k, axis=0)
        edge_normals = np.roll(edge_normals, -k, axis=0)
        pos = np.delete(pos, 1, axis=0)
        edge_normals = edge_normals[1:]
    return pos, edge_normals


def _trace(arcs, old_pos, new_pos, tol=1e-9):
    """Every wavefront vertex sweeps out a skeleton arc as it moves."""
    for a, b in zip(old_pos, new_pos):
        if np.linalg.norm(b - a) > tol:
            arcs.append((a.copy(), b.copy()))


def _all_coincident(pos, tol=1e-7):
    if len(pos) == 0:
        return True
    return bool(np.all(np.hypot(*(pos - pos[0]).T) < tol))


def _emit_degenerate(pos, arcs, nodes, tol=1e-7):
    """Turn a collapsed wavefront into its last skeleton arcs."""
    if len(pos) < 2 or _all_coincident(pos, tol):
        return
    seen = set()
    for i in range(len(pos)):
        a, b = pos[i], pos[(i + 1) % len(pos)]
        if np.linalg.norm(b - a) <= tol:
            continue
        key = tuple(np.round(np.sort([a @ a, b @ b]), 9))
        if key in seen:
            continue
        seen.add(key)
        arcs.append((a.copy(), b.copy()))
        nodes.append(a.copy())
        nodes.append(b.copy())


def _check_wavefront(new_pos, i_edge, guard, tol=1e-6):
    """Catch the two failures that do not announce themselves.

    vertex event   two vertices that are not neighbours in the polygon arrive
                   at the same point. The wavefront is pinching itself in two
                   and, like a split event, needs the loop to be cut and
                   re-joined. An L-shape does this on its very first event.
    escape         a wavefront vertex ends up outside the original polygon,
                   which means the propagation has already gone wrong earlier.
    """
    from scipy.spatial import cKDTree

    n = len(new_pos)
    # A convex polygon legitimately ends with every vertex arriving at the
    # incentre together. That is the terminal collapse, not a pinch.
    if n > 3 and not _all_coincident(new_pos, tol):
        pairs = cKDTree(new_pos).query_pairs(tol, output_type="ndarray")
        for a, b in pairs:
            adjacent = (b - a) % n <= 1 or (a - b) % n <= 1
            expected = {int(i_edge), int((i_edge + 1) % n)}
            if adjacent or {int(a), int(b)} == expected:
                continue
            return ("vertex event", new_pos[a].copy(),
                    f"vertices {a} and {b} meet but are not neighbours")

    if guard is not None:
        boundary, tree, slack = guard
        outside = ~polygon_contains(boundary, new_pos)
        if outside.any():
            # only a real escape if it is out by more than rounding
            d, _ = tree.query(new_pos[outside])
            if (d > slack).any():
                k = int(np.flatnonzero(outside)[np.argmax(d)])
                return ("wavefront escaped the polygon", new_pos[k].copy(),
                        f"vertex {k} is {d.max():.2f}px outside")
    return None


def _split_event_time(pos, vel, normals, offsets, i):
    """Earliest time reflex vertex i reaches any non adjacent moving edge."""
    n = len(pos)
    best, best_j = np.inf, None
    for j in range(n):
        if j in ((i - 1) % n, i % n):
            continue
        denom = normals[j] @ vel[i] - 1.0
        if abs(denom) < 1e-12:
            continue
        t = (offsets[j] - normals[j] @ pos[i]) / denom
        if t <= 1e-9 or t >= best:
            continue
        # the hit has to land on the edge itself, not its extension
        hit = pos[i] + t * vel[i]
        a = pos[j] + t * vel[j]
        b = pos[(j + 1) % n] + t * vel[(j + 1) % n]
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-12:
            continue
        s = ((hit - a) @ ab) / L2
        if -0.02 <= s <= 1.02:
            best, best_j = t, j
    return best, best_j


def propagate(polygon, max_events=5000, on_split="raise"):
    """Shrink the polygon and collect the skeleton arcs.

    on_split:
      "raise"  stop and raise SplitEventError (default, honest)
      "stop"   stop and return what we have, flagged incomplete
      "ignore" carry on regardless, which produces a wrong skeleton on purpose
               so the failure can be plotted
    """
    poly = np.asarray(polygon, dtype=float)
    if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3:
        raise ValueError("need at least 3 vertices")

    from scipy.spatial import cKDTree
    from .polygon import resample_ring

    normals = _inward_normals(poly)
    offsets = np.einsum("ij,ij->i", normals, poly)

    diag = np.hypot(*(poly.max(axis=0) - poly.min(axis=0)))
    guard = (Boundary(poly), cKDTree(resample_ring(poly, 1.0)), 1e-4 * diag)

    pos = poly.copy()
    vel, reflex = _vertex_velocities(poly, normals)
    edge_normals = normals.copy()
    edge_offsets = offsets.copy()

    arcs, nodes, events = [], [], []
    t_now = 0.0

    for _ in range(max_events):
        pos, edge_normals = _collapse_degenerate(pos, edge_normals)
        if len(pos) < 3 or _all_coincident(pos):
            # The wavefront has flattened to a segment (or a point). If it is a
            # segment, that segment is the last piece of skeleton: it is where
            # the two sides of the shape finally met. A rectangle ends here,
            # and forgetting this step costs you its entire central axis.
            _emit_degenerate(pos, arcs, nodes)
            break
        vel, reflex = _vertex_velocities(pos, edge_normals)
        edge_offsets = np.einsum("ij,ij->i", edge_normals, pos) - t_now

        t_edge = _edge_event_times(pos, vel)
        i_edge = int(np.argmin(t_edge))
        t_min = t_edge[i_edge]

        # would a reflex vertex get there first?
        t_split, i_split, j_split = np.inf, None, None
        for i in np.flatnonzero(reflex):
            t, j = _split_event_time(pos, vel, edge_normals, edge_offsets, i)
            if t < t_split:
                t_split, i_split, j_split = t, int(i), j

        if t_split < t_min - 1e-9:
            where = pos[i_split] + t_split * vel[i_split]
            if on_split == "raise":
                raise SplitEventError(
                    f"reflex vertex {i_split} splits the wavefront against "
                    f"edge {j_split} at t={t_split:.3f}; split events are not "
                    f"implemented",
                    time=t_now + t_split, vertex=i_split, position=where)
            if on_split == "stop":
                # the wavefront up to the event is still correct, keep it
                _trace(arcs, pos, pos + t_split * vel)
                events.append((t_now + t_split, "split", where))
                return WavefrontResult(arcs, nodes, events, True,
                                       reason="split event")
            # "ignore": fall through and take the edge event anyway

        if not np.isfinite(t_min):
            return WavefrontResult(arcs, nodes, events, True,
                                   reason="no further edge events")

        # advance the whole wavefront, every vertex traces a skeleton arc
        new_pos = pos + t_min * vel

        # Two more ways the wavefront can go wrong, both of which used to be
        # silent. Checking them here is the difference between "this method
        # does not handle X" and a picture of a skeleton outside its own shape.
        bad = _check_wavefront(new_pos, i_edge, guard)
        if bad is not None:
            kind, where, detail = bad
            if on_split == "raise":
                raise SplitEventError(
                    f"{kind} at t={t_now + t_min:.3f} ({detail}); not handled",
                    time=t_now + t_min, vertex=i_edge, position=where)
            if on_split == "stop":
                _trace(arcs, pos, new_pos)
                events.append((t_now + t_min, kind, where))
                return WavefrontResult(arcs, nodes, events, True, reason=kind)

        _trace(arcs, pos, new_pos)

        t_now += t_min
        node = new_pos[i_edge]
        nodes.append(node.copy())
        events.append((t_now, "edge", node.copy()))

        # Merge the two endpoints of the collapsed edge into one vertex.
        # Two index spaces to keep straight: the *vertex* that goes away is
        # i_edge+1, the *edge* that goes away is i_edge. Rotating the collapsed
        # edge to index 0 first means neither removal ever wraps around, which
        # is otherwise a nasty off-by-one only visible on the last edge.
        new_pos = np.roll(new_pos, -i_edge, axis=0)
        edge_normals = np.roll(edge_normals, -i_edge, axis=0)

        new_pos[0] = node              # vertices 0 and 1 have just met
        pos = np.delete(new_pos, 1, axis=0)
        edge_normals = edge_normals[1:]  # the collapsed edge is gone

        # velocities and offsets get refreshed at the top of the next pass

    return WavefrontResult(arcs, nodes, events, False)


# ---------------------------------------------------------------------------
# Straight skeleton vs medial axis
# ---------------------------------------------------------------------------
# Worth stating plainly, because the paper does not.
#
# The paper's Theorem 5 defines the skeleton as the set of centres of maximal
# disks, which is the medial axis. Its Lemma 9 then shows that where two
# boundary segments meet, the skeleton is their angle bisector. Both are true,
# but the second one is a statement about two *straight* boundary pieces.
#
# At a reflex vertex the nearest boundary feature is the vertex itself, a
# point, not a segment. The set of points equidistant from a point and a line
# is a parabola. So the medial axis of a non-convex polygon contains parabolic
# arcs, while the wavefront construction above only ever produces straight
# segments. They are the same object for convex polygons and different objects
# as soon as there is a concavity, which for a vessel means at every single
# bifurcation.
#
# experiments/03_bisector_vs_medial.py measures how far apart they get.
