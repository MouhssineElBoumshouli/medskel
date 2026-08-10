"""The methods we are comparing against.

Both are pixel based: they work on the mask directly and give back another
binary image, one pixel wide. That difference from a graph is not cosmetic. A
skeleton image has no notion of a branch, so to measure anything you first have
to re-detect junctions and trace paths on the pixel grid, and every branch you
measure is quantised to the pixel lattice.
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize as sk_skeletonize, medial_axis


def thinning(mask):
    """Zhang-Suen style thinning, skimage's default. Topology preserving."""
    return sk_skeletonize(np.asarray(mask) > 0)


def medial_axis_pixels(mask, return_distance=False):
    """Medial axis on the pixel grid, plus the distance transform."""
    return medial_axis(np.asarray(mask) > 0, return_distance=return_distance)


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


def count_pixel_branches(skel):
    """Endpoints, junctions and branch count of a pixel skeleton."""
    g = skeleton_pixel_graph(skel)
    if g.number_of_nodes() == 0:
        return {"endpoints": 0, "junctions": 0, "branches": 0, "n_pixels": 0}

    deg = dict(g.degree())
    endpoints = sum(1 for d in deg.values() if d == 1)
    junctions = sum(1 for d in deg.values() if d >= 3)

    # a branch is a maximal chain between two non-degree-2 nodes
    anchors = {n for n, d in deg.items() if d != 2}
    branches, seen = 0, set()
    for a in anchors:
        for nb in g.neighbors(a):
            if (a, nb) in seen:
                continue
            prev, cur = a, nb
            seen.add((a, nb))
            while g.degree(cur) == 2:
                nxt = [n for n in g.neighbors(cur) if n != prev][0]
                seen.add((cur, nxt))
                seen.add((nxt, cur))
                prev, cur = cur, nxt
            seen.add((cur, prev))
            branches += 1

    return {"endpoints": endpoints, "junctions": junctions,
            "branches": branches, "n_pixels": int(np.count_nonzero(skel))}


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
