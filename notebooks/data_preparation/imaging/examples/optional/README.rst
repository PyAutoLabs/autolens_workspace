The ``imaging/data_preparation`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

Files (Advanced)
----------------

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``mask.py``: Choosing a mask for the analysis.
- ``positions.py``: Marking source-positions on a lensed source such that mass models during a non-linear search are discarded if they do not trace close to one another.
- ``lens_light_centre.py``: Masking the centre of the lens galaxy(s) light to help compose the lens model.
- ``clump_centres.py``: Adding additional clump centres, which add extra light and mass profiles to a lens model.
- ``scaled_dataset.py``: Removing unwanted light from interpolator galaxies and stars in an image.
- ``info.py``: Adding information to the dataset (e.g. redshifts) to aid analysis after lens modeling.