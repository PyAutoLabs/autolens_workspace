The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

Files
-----

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``image``: The image data of the lens, including units.
- ``noise_map``: The corresponding noise map of the image.
- ``psf``:  The Point Spread Function which represents blurring ue to the telescope optics.

Folders
-------

- `optional`: Scripts that are not required for an analysis, but may be useful for certain lenses and analyses.