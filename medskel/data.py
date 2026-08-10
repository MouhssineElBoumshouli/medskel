"""Real images to run on.

Everything here comes from scikit-image's own sample data, so the repo has no
dataset to download and nothing with a licence attached. That was deliberate:
the point is that someone can clone this and get the figures back.

The segmentations are not the contribution and they are not especially good.
They exist to produce realistic binary masks, which is what the skeletonizer
actually consumes. Anywhere the segmentation is shaky, that is worth
remembering when reading the skeleton on top of it.
"""

import warnings

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skimage import data, exposure, filters, measure, morphology, transform


def largest_component(mask):
    lab = measure.label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def retina_vessels(scale=0.5, lo_pct=92, hi_pct=97, closing=4,
                   min_size=150, largest=True):
    """Vessel mask from the fundus photograph shipped with scikit-image.

    Green channel (best vessel contrast in a fundus image), CLAHE, then a
    Frangi vesselness filter and a hysteresis threshold. Hysteresis rather than
    a single threshold because vessels fade distally: a high threshold finds
    the confident vessel cores, the low one lets each vessel grow outwards from
    a core it is connected to, instead of breaking into dashes.

    Returns (mask, green_channel, vesselness).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = data.retina()
        green = transform.rescale(rgb[..., 1], scale, anti_aliasing=True)

        # the circular field of view, eroded so its rim is not read as a vessel
        fov = binary_fill_holes(green > 0.05 * green.max())
        fov = binary_erosion(fov, np.ones((3, 3)), iterations=12)

        eq = exposure.equalize_adapthist(green, clip_limit=0.02)
        vesselness = filters.frangi(eq, sigmas=range(1, 7),
                                    black_ridges=True) * fov

        lo = np.percentile(vesselness[fov], lo_pct)
        hi = np.percentile(vesselness[fov], hi_pct)
        mask = filters.apply_hysteresis_threshold(vesselness, lo, hi)
        mask = morphology.closing(mask, morphology.disk(closing))
        mask = morphology.remove_small_objects(mask, min_size)
        mask = binary_fill_holes(mask)

    if largest:
        mask = largest_component(mask)
    return mask, green, vesselness


def skull_vault(slice_index=4, percentile=93, closing=3, min_size=400):
    """Cross section of the skull from the head scan in scikit-image.

    A different shape class on purpose. The vault is an annulus, so its medial
    axis is a closed loop with no free ends at all, and the clearance radius
    along that loop is half the local bone thickness. Vessels never test that,
    because a vessel tree is a tree.

    Returns (mask, slice_image).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = data.brain()
        img = vol[slice_index].astype(float)

        bone = img > np.percentile(img[img > 0], percentile)
        bone = morphology.closing(bone, morphology.disk(closing))
        bone = morphology.remove_small_objects(bone, min_size)

    mask = largest_component(bone)
    return mask, img
