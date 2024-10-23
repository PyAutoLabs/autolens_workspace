The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoLens** analysis:

Files (Beginner)
----------------

The following scripts are used to prepare the following components of an imaging dataset for analysis:

- ``image.py``: The image data of the lens, including units.
- ``noise_map.py``: The corresponding noise map of the image.
- ``psf.py``:  The Point Spread Function which represents blurring ue to the telescope optics.

Folders (Beginner)
------------------

- `optional`: Scripts that are not required for an analysis, but may be useful for certain lenses and analyses.