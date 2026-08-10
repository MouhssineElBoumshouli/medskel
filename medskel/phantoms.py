"""Synthetic shapes where the true centerline is known.

Real segmentations do not come with a ground truth centerline, so there is no
way to say "this skeleton is 1.8 px off" on a retina image. Phantoms fix that:
we draw the centerline first and grow the shape around it, so the answer is
known before the algorithm runs.

The roughen() function is the important one. It adds noise to the *boundary*
only, leaving the true centerline untouched, which imitates what an imperfect
segmentation actually does to a mask. Any branch that appears because of that
noise is a false branch, and we can count them.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter


def _rasterize(centerline, radius, shape):
    """Union of disks of the given radius along the centerline.

    Only touches a window around each disk, otherwise the vessel tree with a
    few thousand centerline points takes forever.
    """
    h, w = shape
    mask = np.zeros(shape, bool)
    radius = np.broadcast_to(np.asarray(radius, float), (len(centerline),))

    for (cx, cy), r in zip(centerline, radius):
        x0, x1 = int(np.floor(cx - r)), int(np.ceil(cx + r)) + 1
        y0, y1 = int(np.floor(cy - r)), int(np.ceil(cy + r)) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return mask


def straight_tube(shape=(160, 400), radius=18, n=400):
    """Constant width horizontal tube. The easiest possible case."""
    h, w = shape
    x = np.linspace(0.12 * w, 0.88 * w, n)
    y = np.full(n, h / 2.0)
    c = np.column_stack([x, y])
    return _rasterize(c, radius, shape), c


def curved_tube(shape=(300, 500), radius=16, amplitude=60, n=600):
    """A sine shaped tube. Tests that the skeleton follows a curve."""
    h, w = shape
    x = np.linspace(0.10 * w, 0.90 * w, n)
    y = h / 2.0 + amplitude * np.sin(2 * np.pi * (x - x[0]) / (0.8 * w))
    c = np.column_stack([x, y])
    return _rasterize(c, radius, shape), c


def tapered_tube(shape=(220, 500), r0=22, r1=5, n=600):
    """Width shrinks along the tube, like a vessel heading distally."""
    h, w = shape
    x = np.linspace(0.10 * w, 0.90 * w, n)
    y = np.full(n, h / 2.0)
    c = np.column_stack([x, y])
    r = np.linspace(r0, r1, n)
    return _rasterize(c, r, shape), c


def bifurcation(shape=(360, 460), radius=14, angle_deg=35, n=350):
    """A Y. Three centerline arms, one bifurcation.

    This is the phantom that matters for vessels, because the bifurcation is
    the thing the skeleton is supposed to find and it is also the thing that
    a wavefront method has to handle a split event for.
    """
    h, w = shape
    cx, cy = 0.30 * w, h / 2.0
    trunk_x = np.linspace(0.06 * w, cx, n)
    trunk = np.column_stack([trunk_x, np.full(n, cy)])

    arms = []
    L = 0.62 * w
    for sign in (-1, +1):
        a = np.radians(angle_deg) * sign
        t = np.linspace(0, L, n)
        arms.append(np.column_stack([cx + t * np.cos(a), cy + t * np.sin(a)]))

    centerline = [trunk] + arms
    mask = _rasterize(np.vstack(centerline), radius, shape)
    return mask, centerline


def vessel_tree(shape=(520, 520), depth=4, radius0=13, seed=0):
    """A recursive binary tree, a toy version of a vascular tree.

    Not anatomically anything, but it has the property that matters here:
    many bifurcations, and branches that get thin enough that boundary noise
    is comparable to vessel width.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    segments, radii = [], []

    def grow(p, angle, length, r, level):
        q = p + length * np.array([np.cos(angle), np.sin(angle)])
        t = np.linspace(0, 1, max(int(length), 8))
        segments.append(p + np.outer(t, q - p))
        radii.append(np.full(len(t), r))
        if level >= depth:
            return
        spread = np.radians(rng.uniform(20, 38))
        for sign in (-1, +1):
            grow(q, angle + sign * spread, length * rng.uniform(0.62, 0.78),
                 max(r * 0.68, 2.0), level + 1)

    grow(np.array([w / 2.0, h * 0.94]), -np.pi / 2, h * 0.24, radius0, 0)
    centerline = np.vstack(segments)
    mask = _rasterize(centerline, np.concatenate(radii), shape)
    return mask, segments


def roughen(mask, amplitude=2.0, scale=3.0, seed=0):
    """Add correlated noise to the boundary of a mask.

    Works on the signed distance field: perturb it with smoothed noise and
    re-threshold. That keeps the mask a single blob with a wobbly edge instead
    of scattering isolated pixels, which is what a real segmentation error
    looks like.
    """
    from scipy.ndimage import distance_transform_edt

    rng = np.random.default_rng(seed)
    inner = distance_transform_edt(mask)
    outer = distance_transform_edt(~mask)
    sdf = inner - outer

    noise = rng.normal(size=mask.shape)
    noise = gaussian_filter(noise, scale)
    noise /= (noise.std() + 1e-12)

    return (sdf + amplitude * noise) > 0


def largest_component(mask):
    from skimage.measure import label
    lab = label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()
