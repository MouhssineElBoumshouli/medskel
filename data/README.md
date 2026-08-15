# data

Datasets are not committed to this repository. They belong here.

## CHASE_DB1

28 retinal fundus images, each segmented independently by **two** observers.
The second observer is the whole reason this dataset was chosen: the difference
between the two tracings measures how much real experts disagree about where a
vessel boundary sits. `experiments/08_interobserver.py` and
`09_robustness.py` are built on that.

Source, `CHASEDB1.zip`, 2.4 MB:

<https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/>

The zip can sit in this folder as-is, or be extracted here; `medskel/chase.py`
reads either layout. Expected contents are `Image_01L.jpg`,
`Image_01L_1stHO.png`, `Image_01L_2ndHO.png`, and so on for 14 subjects, left
and right eye.

Licence: CC BY 4.0. Source publication:

> M. M. Fraz et al. *An Ensemble Classification-Based Approach Applied to
> Retinal Blood Vessel Segmentation.* IEEE Transactions on Biomedical
> Engineering, 59(9), 2012. doi:10.1109/TBME.2012.2205687

Not redistributed here. The licence would permit it, but a code repository is
the wrong home for someone else's dataset: it bloats every clone and detaches
the files from their source and citation.
