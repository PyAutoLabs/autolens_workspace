The ``chaining`` folder contains examples scripts illustrating the non-linear search chaining feature for dataset datasets.

Files (Advanced)
----------------

The following chaining examples show how to fit a strong lens where:

- ``lens_light_to_total_mass.py``: The lens light model is fitted, followed by a total mass model.
- ``parametric_to_pixelization.py``: A parametric source model is fitted, followed by a pixelized source reconstruction.
- ``sie_to_power_law.py``: A singular isothermal elliptical mass model is fitted, followed by a power-law.
- ``double_einstein_ring.py``: There are two lensed sources at two different redshifts with search chaining.
- ``lens_light_to_light_dark_mass.py``:The lens light model is fitted and then acts as the stellar mass model.
- ``clumps.py``: Chaining is used in combination with the clump API to model small nearby galaxies.
- ``pipeline.py``: Chaining is used to compose a pipeline for five non-linear searches.