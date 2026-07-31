The `pixelization` folder contains example scripts showing how to perform multi-galaxy strong lens analysis using
a pixelized source reconstruction, where the source is reconstructed on a mesh of pixels whose fluxes are solved
via linear algebra.

The scripts apply the pixelization API to a lens with two co-dominant deflectors: the main lens galaxies are
composed individually, each with its own light and mass profile, and the adaptive over-sampling scheme is centred
on every deflector rather than on a single lens.

# Files

- `modeling`: Multi-galaxy lens modeling using a pixelized source reconstruction.
- `fit`: Fit a multi-galaxy lens with a pixelized source without a non-linear search, and inspect the inversion.

# Mesh and Regularization

The examples use a `RectangularAdaptDensity` or `RectangularUniform` mesh with `Constant` regularization. Two
pairings do not work and raise rather than fail silently:

- `AdaptSplit` regularization requires split-cross mappings that the rectangular meshes do not provide. Use a
  Delaunay mesh for the split schemes.
- The `Adapt` regularization schemes require adapt-images, which are derived from a previous model-fit. They are
  used in `multi_galaxy/slam.py`, where an earlier stage provides them.

# Not yet written

The following are present in `imaging/features/pixelization` but not yet here: `adaptive`, `delaunay`,
`cpu_fast_modeling`, `slam`, `source_science` and `plot`. Until they land, those scripts apply with the single
lens galaxy swapped for the `lens_0`, `lens_1`, ... loop of this package.

# Related

- `multi_galaxy/modeling`: the multi-galaxy lens model these scripts extend.
- `imaging/features/pixelization`: the galaxy-scale walkthrough, with the full mesh and regularization API.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
