The `pixelization` folder contains example scripts showing how to perform multi-galaxy strong lens analysis using
a pixelized source reconstruction, where the source is reconstructed on a mesh of pixels whose fluxes are solved
via linear algebra.

The scripts apply the pixelization API to a lens with two co-dominant deflectors: the main lens galaxies are
composed individually, each with its own light and mass profile, and the adaptive over-sampling scheme is centred
on every deflector rather than on a single lens.

# Files

- `modeling`: Multi-galaxy lens modeling using a pixelized source reconstruction.
- `fit`: Fit a multi-galaxy lens with a pixelized source without a non-linear search, and inspect the inversion.
- `likelihood_function`: Step-by-step walkthrough of the pixelized log likelihood function.
- `adaptive`: The adaptive mesh and adaptive regularization, set up by chaining four searches.
- `delaunay`: A Delaunay triangulation mesh, and the split regularization schemes it supports.
- `cpu_fast_modeling`: Fast pixelized modeling on the CPU, using sparse operators instead of JAX.
- `source_science`: Source flux, magnification and errors measured from the reconstruction.
- `slam`: The SLaM pipeline, with the pixelization choices its SOURCE PIX stages make written out.
- `plot`: Plotting a pixelized fit's inversion, mappers and mesh grids.

# Mesh and Regularization

The rectangular meshes (`RectangularUniform`, `RectangularBilinearAdaptDensity`, `RectangularBilinearAdaptImage`) are used by
`modeling`, `fit`, `adaptive`, `cpu_fast_modeling` and `slam`; `delaunay` uses the `Delaunay` mesh.

Two constraints govern which regularization pairs with which mesh, and both fail loudly rather than silently:

- The split schemes (`ConstantSplit`, `AdaptSplit`) require split-cross mappings that the rectangular meshes do
  not provide, and raise a `PixelizationException` against them. Use a Delaunay mesh — see `delaunay`.
- The `Adapt` schemes require adapt-images, which are derived from a previous model-fit. `adaptive` and `slam`
  produce their own within the script; a standalone fit uses `Constant` instead.

# Related

- `multi_galaxy/modeling`: the multi-galaxy lens model these scripts extend.
- `multi_galaxy/slam`: the baseline SLaM pipeline, which this folder's `slam` diffs against.
- `imaging/features/pixelization`: the galaxy-scale walkthrough, with the full mesh and regularization API.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
