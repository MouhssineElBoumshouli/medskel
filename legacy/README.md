# legacy

This is the version I wrote first, in early 2025, kept on purpose.

`first_attempt.py` does not work. It intersects the bisectors of *every pair*
of polygon vertices at once, whereas the paper says to intersect the bisectors
of neighbouring vertices and take the pair that meets first, then repeat. So it
never propagates anything, and what comes out is a cloud of intersection points
with no relation to the skeleton.

`tt.jpg` is the image I was testing on: a dark blob on a white background. The
loader thresholds at 127 and keeps the *bright* side, so it was skeletonizing
the white background and not the blob. That is why `figures/00_first_attempt.png`
shows a diamond floating inside a rectangle: the rectangle is the image border.

Both mistakes are described in the main README. Kept here so the claim that
they happened is checkable rather than just asserted.
