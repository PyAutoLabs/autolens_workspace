The `interferometer/features` folder contains example scripts showing how to fit a lens model to interferometer data using
different **PyAutoLens** features.

The scripts in this folder are all recommended, as they provide tools which make lens modeling more reliable and efficient.
Most users will benefit from these features irrespective of the quality of their data, complexity of their lens model
and scientific topic of study.

# Folders

The following example scripts illustrating lens modeling where:

- `pixelization`: The source is reconstructed using an adaptive rectangular or Delaunay mesh
- `datacube`: Spectral-line data cubes (e.g. ALMA CO cubes), fitting every channel simultaneously with a shared lens model and a per-channel source.
- `extra_galaxies`: Modeling which account for the light and mass of extra nearby galaxies.
- `advanced`: Advanced features for expert users, for example shapelets, potential correction (gravitational imaging) and dark matter subhalo detection.

# Notes

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
