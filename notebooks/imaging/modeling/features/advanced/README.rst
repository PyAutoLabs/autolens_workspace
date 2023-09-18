The ``modeling/features`` folder contains example scripts showing how to fit a lens model to imaging data using
different **PyAutoLens** features.

The scripts in this folder are advanced, and generally provide more niche functionality which will only be useful
for specific scientific topics.

Files (Beginner)
----------------

The following example scripts illustrating lens modeling where:

- ``shapelets.py``: The source (or lens) is reconstructed using shapelet basis functions.
- ``stellar_dark_mass.py``: The lens galaxy's mass is decomposed into stellar and dark matter mass components.
- ``clumps.py``: The model includes additional nearby galaxies as light and mass profiles using PyAutoLens's clump API.
- ``operated_light_profiles.py``: There are light profiles which are assumed to already be convolved with the instrumental PSF (e.g. point sources), commonly used for modeling bright AGN in the centre of a galaxy.
- ``double_einstein_ring.py``: The lens is a double Einstein ring system with two lensed sources at different redshifts.

Notes
-----

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.