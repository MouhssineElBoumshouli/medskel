# data

Nothing in this folder is committed. The experiments that need a dataset expect
you to put it here yourself.

## CHASE_DB1

Used by `experiments/08_interobserver.py`. 28 retinal images, each segmented
independently by **two** human observers, which is the whole reason this
dataset was chosen: the difference between the two observers is a real
measurement of how much experts disagree about where a vessel boundary is.

Download `CHASEDB1.zip` (2.4 MB) from Kingston University's research portal:

<https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/>

The direct file link is `/files/40659508/CHASEDB1.zip` on that domain, but the
server sits behind a bot check, so it has to be fetched with a browser rather
than `curl`.

Then either drop the zip in here as `data/CHASEDB1.zip`, or extract it to
`data/CHASEDB1/`. The loader handles both.

Expected contents: `Image_01L.jpg`, `Image_01L_1stHO.png`,
`Image_01L_2ndHO.png`, and so on for 14 subjects, left and right eye.

**Licence:** CC BY 4.0. Cite the source paper:

> M. M. Fraz et al., *An Ensemble Classification-Based Approach Applied to
> Retinal Blood Vessel Segmentation*, IEEE Transactions on Biomedical
> Engineering, 59(9), 2012. doi:10.1109/TBME.2012.2205687

## Why the data is not in the repo

CC BY would permit redistributing it, but a code repository is the wrong place
for someone else's dataset — it bloats the clone and detaches the files from
their source and citation. Downloading it takes ten seconds.
