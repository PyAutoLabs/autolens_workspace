The `multi_gaussian_expansion` folder contains example scripts showing how to fit `Interferometer` data
using a **multi-Gaussian expansion (MGE)** — a decomposition of a galaxy's light into many linear Gaussian
components whose `intensity` values are solved for analytically via a linear inversion.

A key difference vs the imaging MGE examples: those fit the *lens* galaxy's complex morphology with the MGE.
For interferometer data the lens light is omitted (no detection in mm/sub-mm), so the MGE is instead
applied to the **source** galaxy. The MGE source captures the asymmetric morphology of typical
sub-mm/radio-selected lensed sources better than a single Sersic, while keeping the non-linear parameter
space small (only Gaussian centres and ellipticities are free).

This used to be impractical because every likelihood evaluation had to NUFFT each of the ~15-100 Gaussians
in the basis, and prior NUFFT backends were not JAX-friendly. With nufftax
(https://github.com/GragasLab/nufftax) the full MGE basis is transformed inside the same jit/vmap pipeline
as the rest of the model, so MGE fits to visibilities are now routine even at ALMA-class visibility counts.

# Files

- `modeling`: Lens modeling of an `Interferometer` dataset with an MGE source bulge built from 30
  linear `Gaussian` profiles arranged in two groups of 15 (each group shares a centre and ell_comps;
  sigmas are fixed to log-spaced values).
- `fit`: Fit a known-parameter MGE source and inspect the per-Gaussian solved-for `intensity` values.
- `likelihood_function`: Step-by-step walkthrough of the visibility-plane MGE linear inversion — each
  Gaussian is one column of the real-space mapping matrix, NUFFT'd to the uv-plane, then the standard
  `D`/`F` solve over the joint complex visibility data.
- `slam`: SLaM pipeline (SOURCE LP → SOURCE PIX 1 → SOURCE PIX 2 → MASS TOTAL) using an MGE source in
  the SOURCE LP stage via `al.model_util.mge_model_from`. SOURCE LP runs on `TransformerNUFFT`; the
  pixelized stages use the same `TransformerNUFFT` plus `apply_sparse_operator` (FFT-based W̃ precision
  matrix).

# Results

These scripts only give a brief overview of how to analyse and interpret the results of a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.

# Imaging Equivalent

For the CCD-imaging version of these scripts (MGE on the lens galaxy, not the source), see
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion`.
