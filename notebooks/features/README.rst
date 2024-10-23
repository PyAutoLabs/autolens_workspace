The ``modeling/features`` folder contains example scripts showing how to fit a lens model to imaging data using
different **PyAutoLens** features.

The scripts in this folder are all recommend, as they provide tools which make lens modeling more reliable and efficient.
Most users will benefit from these features irrespective of the quality of their data, complexity of their lens model
and scientific topic of study.

Files (Beginner)
----------------

The following example scripts illustrating lens modeling where:

- ``no_lens_light.py``: The foreground lens's light is not present in the data and thus omitted from the model.
- ``extra_galaxies.py``: Modeling which account for the light and mass of extra nearby galaxies.
- ``linear_light_profiles.py``: The model includes light profiles which use linear algebra to solve for their intensity, reducing model complexity.
- ``multi_gaussian_expansion.py``: The lens (or source) light is modeled as ~25-100 Gaussian basis functions
- ``pixelization.py``: The source is reconstructed using an adaptive Delaunay or Voronoi mesh.

Notes
-----

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at ``autolens_workspace/*/imaging/results``.