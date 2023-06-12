The ``modeling/features`` folder contains example scripts showing how to fit a lens model to imaging data using
different **PyAutoLens** features:

Files (Beginner)
----------------

The following example scripts illustrating lens modeling where:

- ``no_lens_light.py``: The foreground lens's light is not present in the data and thus omitted from the model.
- ``linear_light_profiles.py``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``multi_gaussian_expansion.py``: The lens (or source) light is modeled as ~25-100 Gaussian basis functions to capture asymmetric morphological features.
- ``shapelets.py``: The source (or lens) is reconstructed using shapelet basis functions.
- ``pixelization.py``: The source is reconstructed using an adaptive Voronoi mesh.
- ``stellar_dark_mass.py``: The lens galaxy's mass is decomposed into stellar and dark matter mass components.
- ``clumps.py``: The model includes additional nearby galaxies as light and mass profiles using PyAutoLens's clump API.
- ``operated_light_profiles.py``: There are light profiles which are assumed to already be convolved with the instrumental PSF (e.g. point sources), commonly used for modeling bright AGN in the centre of a galaxy.
- ``double_einstein_ring.py``: The lens is a double Einstein ring system with two lensed sources at different redshifts.

Notes
-----

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.