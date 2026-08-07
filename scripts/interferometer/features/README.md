The `interferometer/features` folder contains example scripts showing how to fit a lens model to interferometer data using
different **PyAutoLens** features.

The scripts in this folder are all recommended, as they provide tools which make lens modeling more reliable and efficient.
Most users will benefit from these features irrespective of the quality of their data, complexity of their lens model
and scientific topic of study.

# Folders

The following example scripts illustrating lens modeling where:

- `pixelization`: The source is reconstructed using an adaptive rectangular or Delaunay mesh
- `datacube`: Spectral-line data cubes (e.g. ALMA CO cubes), fitting every channel simultaneously with a shared lens model and a per-channel source.
- `linear_light_profiles`: Light profiles whose `intensity` is solved for analytically via a linear inversion instead of being a free parameter of the non-linear search.
- `multi_gaussian_expansion`: The source's light decomposed into many linear Gaussian components (MGE), whose intensities are solved for analytically.
- `scaling_relation`: A population of foreground galaxies included by tying their masses to the main lens's through a luminosity scaling relation, rather than freeing each one.
- `extra_galaxies`: Modeling which account for the light and mass of extra nearby galaxies.
- `advanced`: Advanced features for expert users, for example shapelets, potential correction (gravitational imaging) and dark matter subhalo detection.

# Notes

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
