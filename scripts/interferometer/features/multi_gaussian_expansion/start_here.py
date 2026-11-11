"""
Features: Multi Gaussian Expansion
==================================

A multi Gaussian expansion (MGE) represents a galaxy's light as the sum of many Gaussian profiles. Each
Gaussian describes emission on a particular spatial scale; together, they can provide a flexible model of the
galaxy.

This introduction explains how an MGE works, why it can be useful, and what to consider before using one. The
neighbouring examples show how to construct an MGE, fit it to data and inspect the results.

__Why Use Many Gaussians?__

A simple light profile, such as an elliptical Sersic, describes a galaxy using a small number of parameters.
This makes it useful and relatively easy to interpret, but its restricted shape may leave structured residuals
when fitted to a more complex galaxy.

For example, a galaxy's ellipticity or orientation may change with radius. A single elliptical profile cannot
describe all of those changes.

An MGE combines Gaussians with different widths and, depending on the model, different shapes. Narrow
Gaussians describe central structure, while broader Gaussians describe more extended emission. Their
combination can represent radial changes that a single profile misses.

The flexibility depends on how the components are arranged. Giving every Gaussian the same centre and shape
still imposes restrictions; adding more Gaussians does not automatically describe every possible galaxy
morphology.

__How Are the Intensities Fitted?__

For fixed Gaussian widths, centres and shapes, the model image depends linearly on the Gaussian intensities.
We can therefore solve for those intensities during each likelihood evaluation, instead of treating them as
separate parameters in the non-linear search.

The fitting process is:

1. The non-linear search proposes the model's remaining parameters.
2. The corresponding Gaussian images are calculated, including the instrumental response and, for source light, gravitational lensing.
3. A linear solve determines the Gaussian intensities.
4. The resulting model is compared with the data.

An MGE containing 60 Gaussians therefore does not require the search to sample 60 additional intensity
parameters.

The number of non-linear parameters depends on the choices you make: which widths are fixed, which centres are
shared, and which shapes are allowed to vary. The modeling example explains its choices alongside the code.

__Positive Intensities__

The examples constrain Gaussian intensities to be non-negative.

Without this constraint, overlapping Gaussians can acquire large positive and negative intensities that cancel
one another. Such a combination may fit the data, but its individual components no longer represent positive
contributions to the galaxy's light.

Non-negative intensities avoid this cancellation. They do not, however, guarantee that the model is adequate:
you should still inspect the residuals and consider whether the chosen components can describe the observed
structure.

__Source Light__

An MGE can also describe a lensed source. Solving its intensities and fixing a suitable range of Gaussian
widths can reduce the number of parameters explored by the non-linear search.

However, the same modelling restrictions apply. A common-centre MGE may be poorly suited to separated
star-forming clumps or multiple background galaxies. Changing the basis can add flexibility, but it also
changes the model you are asking the data to constrain.

If your aim is to reconstruct detailed, irregular source morphology, read the pixelization introduction as
well. Follow the available source examples below to see how an MGE source is constructed.

__Computational and Scientific Trade-offs__

Evaluating many Gaussian components and solving their intensities adds work to each likelihood calculation. On
the other hand, reducing the number of sampled parameters can make the non-linear search easier. The balance
depends on the dataset and model; an MGE is not guaranteed to make every fit faster.

An MGE is also less directly interpretable than a single Sersic profile. An individual Gaussian is usually a
component of the mathematical description, rather than a separate physical structure in the galaxy. Quantities
such as total luminosity or a half-light radius should be derived from the combined light distribution.

__In This Folder__

These examples fit visibilities. The image-plane light prediction is transformed into the measured
Fourier-domain quantities; a dirty image is an inspection product, not a substitute for that likelihood. Keep
the visibility sampling, noise and model-specific response choices in view when adapting an imaging example.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/modeling.ipynb) — Fit the model with a non-linear search.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/likelihood_function.ipynb) — Follow the likelihood calculation.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/slam.ipynb) — Follow the staged SLaM fit.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
