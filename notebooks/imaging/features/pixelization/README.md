The `pixelization` folder contains example scripts showing how to perform analysis using a pixelized source reconstruction.

# Files

The following example scripts illustrating lens modeling where:

- `modeling`: Lens modeling using a pixelized source reconstruction.
- `fit`: Fit a pixelized source and compute quantities like the residuals, chi squared and likelihood.
- `plot`: How to plot fits which reconstruct the source galaxy using a pixelization.
- `likelihood_function`: A step-by-step guide of the pixelized source likelihood function.
- `cpu_fast_modeling`: How to speed up pixelized source modeling using CPUs ,if you do not have access to modern GPUs.
- `source_science`: Performing source science calculations like the unlensed source's total flux and magnification.
- `adaptive`: Advanced pixelization features which adapt the mesh and regularization to the source being reconstructed.
- `slam`: Using the Source, Light and Mass (SLAM) pipeline to perform lens modeling using pixelized source reconstruction.
- `delaunay`: Using a Delaunay mesh (instead of a rectangular mesh) for the source reconstruction.

# Rectangular Mesh Variants

The default adaptive rectangular mesh is `RectangularBilinearAdaptDensity` (with `RectangularBilinearAdaptImage`
its adapt-image counterpart): it warps the grid via the empirical rank CDF of the traced points — no extra
parameters and the fastest rectangular mesh on CPUs. The advanced `RectangularRTUAdaptDensity` /
`RectangularRTUAdaptImage` meshes use a smooth kernel-density CDF instead — the ray-guided transformed uniform
(RTU) grid formulation of Enzi et al. (2026), https://arxiv.org/abs/2606.30620, which should be cited when using
them. Use RTU on GPUs, for gradient-based (JAX) samplers (the Bilinear likelihood has zero gradients at the
default `over_sample_size_pixelization=1` — set it >= 4 or use RTU), and for interferometer gradient fitting.

# Results

These scripts only give a brief overview of how to analyse and interpret the results a lens model fit.

A full guide to result analysis is given at `autolens_workspace/*/guides/results`.
