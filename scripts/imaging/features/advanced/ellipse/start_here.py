"""
Advanced Features: Ellipse Fitting
=================================

Measure the foreground lens galaxy's light morphology before choosing a lens-light model. Ellipse fitting
traces how the ellipticity, position angle and centre of its isophotes change with radius, revealing twists,
centre shifts and departures from elliptical symmetry that a single Sersic profile may not capture.

__Scientific Use Cases__

Use these measurements to assess whether the lens light needs a more flexible representation, such as a
multi Gaussian expansion, and to characterize asymmetric or interacting lens galaxies. The measured light
structure can motivate tests of lens models, but it does not directly determine the mass distribution.

[Amvrosiadis et al. (2024)](https://arxiv.org/abs/2407.12983) use the m=1 multipole to characterize lopsided
isophotes in early-type galaxies and discuss their implications for strong-lens mass models.

__How It Works__

At a chosen radius, an ellipse defines sample positions in the image. PyAutoGalaxy interpolates the image
and noise map at those positions, then compares the sampled brightness with its mean around the ellipse.
Fitting the centre and elliptical components and repeating at multiple radii measures the observed isophotal
structure without assuming a radial light-profile law. Multipoles extend the ellipse to describe departures
from elliptical symmetry.

The examples fit observed isophotes without PSF convolution of a light model. Interpret measurements near
the PSF scale with care. Mask lensed arcs, companions and other contaminating light when the aim is to
measure the foreground galaxy's morphology.

__Worked Examples in PyAutoGalaxy__

Ellipse fitting is implemented in PyAutoGalaxy's `autogalaxy.ellipse` subpackage, with `Ellipse`, `FitEllipse`
and `AnalysisEllipse` available through its main API. Use the maintained PyAutoGalaxy examples:

1. [Fit](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/fit.ipynb):
   understand interpolation, residuals, chi-squared and the likelihood for an ellipse.
2. [Modeling](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/modeling.ipynb):
   infer ellipse parameters with a non-linear search and repeat at multiple radii.
3. [Multipoles](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/multipoles.ipynb):
   fit deviations from elliptical isophotes.

Run those notebooks in the autogalaxy_workspace, where their configurations and simulated datasets are
provided. This AutoLens page is an introduction and route to those examples, not a separate implementation.

__A Different Use of Ellipse Geometry__

The [Witt-Wynne guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/misc/witt_wynne.ipynb)
uses ellipse geometry in a specialized approximation for lensed point-source positions. It addresses a
different problem from measuring a galaxy's light isophotes.
"""
