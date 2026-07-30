The `linear_light_profiles` folder contains example scripts showing how to fit `Interferometer` data using
**linear light profiles**, whose `intensity` is solved for analytically via a linear inversion instead of
being a free parameter of the non-linear search.

For interferometer data this is now practical thanks to the JAX-native NUFFT `nufftax`
(https://github.com/GragasLab/nufftax), which evaluates the image-to-uv Fourier transform of every basis
component inside the same jit/vmap pipeline as the rest of the model. Older interferometer guidance that
described light-profile fitting as "slow" pre-dates this change.

# Files

- `modeling`: Lens modeling of an `Interferometer` dataset with a linear `SersicCore` source.
- `fit`: Fit a linear light profile model to interferometer data and inspect the solved-for intensities.
- `likelihood_function`: A step-by-step walkthrough of the linear-light-profile interferometer likelihood
  function (NUFFT of each linear basis image, mapping/curvature matrices, $\chi^2$ in the visibility plane).
- `slam`: SLaM SOURCE LP / SOURCE PIX / MASS TOTAL pipeline using a linear `Sersic` source in the
  initialization stage. Light-profile fitting runs on `TransformerNUFFT` and the pixelized stages use the
  same `TransformerNUFFT` plus `apply_sparse_operator` (FFT-based W̃ precision matrix).

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts, see `autolens_workspace/*/imaging/features/linear_light_profiles`.
