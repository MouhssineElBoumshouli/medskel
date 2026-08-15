"""The methods we are comparing against.

Both are pixel based: they work on the mask directly and give back another
binary image, one pixel wide. That difference from a graph is not cosmetic. A
skeleton image has no notion of a branch, so to measure anything you first have
to re-detect junctions and trace paths on the pixel grid, and every branch you
measure is quantised to the pixel lattice.
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize as sk_skeletonize


def thinning(mask):
    """Zhang-Suen style thinning, skimage's default. Topology preserving."""
    return sk_skeletonize(np.asarray(mask) > 0)


def skeleton_pixel_graph(skel):
    """Graph of a pixel skeleton, 8-connected. Needed to count branches.

    This function is the point of the comparison, more than its output is:
    getting from a skeleton image to something countable takes this much work,
    and the Voronoi version hands it to you.
    """
    skel = np.asarray(skel) > 0
    ys, xs = np.nonzero(skel)
    index = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}

    g = nx.Graph()
    for i, (y, x) in enumerate(zip(ys, xs)):
        g.add_node(i, pos=np.array([float(x), float(y)]))

    for (y, x), i in index.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index.get((y + dy, x + dx))
                if j is not None and j > i:
                    g.add_edge(i, j, length=float(np.hypot(dy, dx)))
    return g


def merge_junction_clusters(g):
    """Collapse touching junction pixels into a single junction node.

    Three lines cannot meet at one point on a pixel grid. Thinning leaves a
    little knot of two to five mutually adjacent pixels instead, each of which
    has three or more neighbours. Counted naively, that knot reads as several
    branches one pixel long.

    It is not a small effect and it is not fair to the baseline: on a clean
    Y-shaped phantom, naive counting reports 8 branches for a shape with 3, and
    all 5 extras are 1px long and sitting on top of each other at the single
    junction. Merging the knot first makes a "branch" mean the same thing for a
    pixel skeleton as it does for the Voronoi graph, which is the only way the
    comparison in experiments/03_noise.py means anything.
    """
    g = g.copy()
    junction_nodes = [n for n, d in g.degree() if d >= 3]
    if not junction_nodes:
        return g

    for cluster in list(nx.connected_components(g.subgraph(junction_nodes))):
        cluster = list(cluster)
        keep = cluster[0]
        for other in cluster[1:]:
            if other in g and keep in g:
                g = nx.contracted_nodes(g, keep, other, self_loops=False)
    return g


def branch_paths(skel, merge_junctions=True):
    """Every branch of a pixel skeleton, as a list of pixel coordinates.

    A branch is a maximal chain running between two nodes that are not simple
    pass-through pixels (i.e. endpoints or junctions).
    """
    g = skeleton_pixel_graph(skel)
    if g.number_of_nodes() == 0:
        return [], g
    if merge_junctions:
        g = merge_junction_clusters(g)

    pos = nx.get_node_attributes(g, "pos")
    anchors = {n for n, d in g.degree() if d != 2}
    paths, seen = [], set()

    for a in anchors:
        for nb in g.neighbors(a):
            if (a, nb) in seen:
                continue
            prev, cur, path = a, nb, [a, nb]
            seen.add((a, nb))
            while g.degree(cur) == 2:
                nxt = [n for n in g.neighbors(cur) if n != prev][0]
                seen.add((cur, nxt))
                seen.add((nxt, cur))
                prev, cur = cur, nxt
                path.append(cur)
            seen.add((cur, prev))
            paths.append(np.array([pos[n] for n in path]))

    return paths, g


def count_pixel_branches(skel, merge_junctions=True):
    """Endpoints, junctions and branch count of a pixel skeleton.

    merge_junctions=False reproduces the naive count, kept so the size of the
    artefact can be shown rather than just asserted.
    """
    if np.count_nonzero(skel) == 0:
        return {"endpoints": 0, "junctions": 0, "branches": 0, "n_pixels": 0}

    paths, g = branch_paths(skel, merge_junctions=merge_junctions)
    deg = dict(g.degree())
    return {"endpoints": sum(1 for d in deg.values() if d == 1),
            "junctions": sum(1 for d in deg.values() if d >= 3),
            "branches": len(paths),
            "n_pixels": int(np.count_nonzero(skel))}


def polyline_vertices(skel, simplify=0.25):
    """How many points it takes to store a pixel skeleton as polylines.

    The fair counterpart to Skeleton.n_polyline_vertices(). Comparing our
    simplified polylines against the baseline's raw pixel count would be
    measuring our simplification step against their lack of one, which says
    nothing about the two skeletons.
    """
    from .voronoi import _rdp_indices

    paths, _ = branch_paths(skel)
    return int(sum(len(_rdp_indices(p, simplify)) for p in paths))


def prune_pixel_skeleton(skel, min_length=5):
    """Remove short spurs from a pixel skeleton.

    A fixed length threshold, which is the usual way it is done and also the
    reason it is awkward: the right threshold on a wide structure is the wrong
    one on a thin structure, and here there is no radius to scale it by.
    """
    skel = np.asarray(skel) > 0
    g = skeleton_pixel_graph(skel)
    pos = nx.get_node_attributes(g, "pos")

    changed = True
    while changed:
        changed = False
        for leaf in [n for n, d in g.degree() if d == 1]:
            if leaf not in g:
                continue
            path, prev, cur = [leaf], None, leaf
            while True:
                nbrs = [n for n in g.neighbors(cur) if n != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
                if g.degree(cur) != 2:
                    break
            if g.degree(path[-1]) > 2 and len(path) - 1 < min_length:
                g.remove_nodes_from(path[:-1])
                changed = True

    out = np.zeros_like(skel)
    for n in g:
        x, y = pos[n]
        out[int(y), int(x)] = True
    return out
