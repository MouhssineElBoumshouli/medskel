"""Loader for CHASE_DB1, the two-observer retinal dataset.

The point of this dataset here is not the images. It is that every image was
segmented independently by two people, so the difference between their two
masks is a measurement of how much real experts disagree about where a vessel
boundary sits. That is the perturbation medskel claims to be robust to,
measured rather than invented.

The files are named Image_01L.jpg, Image_01L_1stHO.png, Image_01L_2ndHO.png,
for 14 subjects, left and right eye. "HO" is human observer.

The dataset is not fetched automatically; see data/README.md for the source.
A library quietly downloading several megabytes on import is a bad habit, and
it hides where the data came from.
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_NAME = "CHASEDB1.zip"

_PATTERN = re.compile(
    r"Image_(\d{2}[LR])(?:_(1st|2nd)HO)?\.(jpg|png)$", re.IGNORECASE)


class ChaseNotAvailable(FileNotFoundError):
    """Raised with instructions rather than a bare path error."""

    def __init__(self):
        super().__init__(
            f"CHASE_DB1 not found in {DATA_DIR}.\n"
            f"Download CHASEDB1.zip (2.4 MB, CC BY 4.0) from\n"
            f"  https://researchinnovation.kingston.ac.uk/en/datasets/"
            f"chasedb1-retinal-vessel-reference-dataset-4/\n"
            f"and put it in {DATA_DIR} (zip or extracted). "
            f"See data/README.md.")


def _binarise(arr):
    """Observer masks are saved as images; anything non-black is vessel."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
    return arr > (0.5 * arr.max() if arr.max() > 1 else 0.5)


def _read_image(data):
    from skimage import io as skio
    return skio.imread(io.BytesIO(data))


def available():
    """Is the dataset present, in either form?"""
    if (DATA_DIR / ZIP_NAME).exists():
        return True
    return any(DATA_DIR.glob("**/Image_*_1stHO.png"))


def _entries_from_zip(path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = _PATTERN.search(Path(name).name)
            if m:
                yield m, z.read(name)


def _entries_from_dir(root):
    for p in sorted(root.glob("**/Image_*")):
        m = _PATTERN.search(p.name)
        if m:
            yield m, p.read_bytes()


def load_cases():
    """Every case as (name, image, observer1_mask, observer2_mask).

    Cases missing either observer are skipped and reported by load_summary().
    """
    if not available():
        raise ChaseNotAvailable()

    zip_path = DATA_DIR / ZIP_NAME
    source = (_entries_from_zip(zip_path) if zip_path.exists()
              else _entries_from_dir(DATA_DIR))

    bundles = {}
    for m, data in source:
        case, observer = m.group(1).upper(), m.group(2)
        slot = "image" if observer is None else observer.lower()
        bundles.setdefault(case, {})[slot] = data

    cases = []
    for case in sorted(bundles):
        b = bundles[case]
        if "1st" not in b or "2nd" not in b:
            continue
        cases.append({
            "name": case,
            "image": _read_image(b["image"]) if "image" in b else None,
            "obs1": _binarise(_read_image(b["1st"])),
            "obs2": _binarise(_read_image(b["2nd"])),
        })

    if not cases:
        raise ChaseNotAvailable()
    return cases


def dice(a, b):
    """Overlap between two masks. 1.0 is identical, 0.0 is disjoint.

    Reported before anything else, because it says whether the two observers
    are disagreeing about vessel *edges* (which is what we want to study) or
    about which vessels exist at all (which would confound the whole thing).
    """
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 1.0


def load_summary():
    """Quick description of what was found, for a sanity check."""
    cases = load_cases()
    rows = []
    for c in cases:
        rows.append({
            "name": c["name"],
            "shape": tuple(c["obs1"].shape),
            "obs1_px": int(c["obs1"].sum()),
            "obs2_px": int(c["obs2"].sum()),
            "dice": dice(c["obs1"], c["obs2"]),
        })
    return rows
