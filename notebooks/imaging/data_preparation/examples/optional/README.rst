The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

Files
-----

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``mask``: Choosing a mask for the analysis.
- ``positions``: Marking source-positions on a lensed source such that mass models during a non-linear search are discarded if they do not trace close to one another.
- ``lens_light_centre``: Masking the centre of the lens galaxy(s) light to help compose the lens model.
- ``extra_galaxies_centres``: Adding additional extra galaxy centres, which add extra light and mass profiles to a lens model.
- ``mask_extra_galaxies``: Removing unwanted light from interpolator galaxies and stars in an image.
- ``info``: Adding information to the dataset (e.g. redshifts) to aid analysis after lens modeling.