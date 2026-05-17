The `shapelets` folder contains example scripts showing how to fit `Interferometer` data using a
**shapelet decomposition** of the source galaxy's light: a polar (Gauss-Hermite) basis whose `intensity`
values are solved for analytically via a linear inversion.

Shapelet fits to interferometer data were previously impractical because every iteration has to NUFFT each
basis component, and prior NUFFT backends were not JAX-friendly. With nufftax
(https://github.com/GragasLab/nufftax) the full shapelet basis is transformed inside the same jit/vmap
pipeline as the rest of the model, so shapelets-on-visibilities is now practical at any visibility count.

Lens light is omitted (interferometer convention). Shapelets are applied to the source galaxy.

# Files

- `modeling`: Lens modeling of an `Interferometer` dataset with a polar shapelet source bulge.
- `fit`: Fit a known-parameter shapelet source and inspect the per-shapelet solved-for `intensity`
  values.

# Positive-Negative Solver

Unlike MGE or linear-light-profile-Sersic sources, shapelets **require** the positive-negative linear
algebra solver. Shapelets are a mathematical basis (not physically motivated profiles), and their ability
to decompose galaxy morphology depends on being able to mix positive and negative basis-function
amplitudes. The `Settings(use_positive_only_solver=False)` toggle is therefore set in the analysis.

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts, see
`autolens_workspace/scripts/imaging/features/advanced/shapelets`.
