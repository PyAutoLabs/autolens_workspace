"""
Features: Multi Gaussian Expansion
==================================

In a strong-lens image, light from two galaxies is superimposed: the foreground lens galaxy and the distant
source, stretched into arcs by gravitational lensing. Modelling their light helps us separate the two and
infer the properties of the lens and source.

A multi Gaussian expansion (MGE) represents a galaxy's light as the sum of many Gaussian profiles. Each
Gaussian describes emission on a particular spatial scale; together, they can provide a flexible model of the
galaxy.

In PyAutoLens, there are two main reasons to use an MGE:

- **Model the foreground lens light:** capture structure that a simple light profile misses, helping separate
  bright foreground emission from overlapping lensed arcs.
- **Model the background source:** describe its unlensed light distribution, predict the arcs through a lens
  mass model, and measure quantities such as source flux and magnification.

![MGE illustration: separate foreground lens light and model the background source](../../../../images/features/mge_illustration.png)

Illustration, not observational data or a fitted result. Amber represents foreground lens light and cyan
represents source light. The source model is mapped into arcs by the lens mass model; subtracting foreground
light alone does not reconstruct the unlensed source. Gaussian components are mathematical building blocks,
not separate physical galaxies. The components are fitted jointly; the arrows illustrate their relationship.

__Foreground Lens Light__

In strong-lens imaging, foreground galaxy light can overlap the lensed arcs. Residuals from an inadequate
foreground model can then interfere with the reconstruction of the background source.

An MGE provides a flexible description of the foreground light. This can help separate it from the arcs,
particularly when a simpler light profile leaves coherent residuals.

For a scientific application, see [He et al. (2024), Unveiling Lens Light Complexity with A Novel
Multi-Gaussian Expansion Approach for Strong Gravitational Lensing](https://arxiv.org/abs/2403.16253).
They apply an MGE lens-light model to 38 HST SLACS lenses, combining it with a pixelized source reconstruction.
This is an example of using MGE for the foreground light; it does not require the source to use an MGE too.

The lens and source should still be considered together. A flexible foreground model can absorb emission that
belongs to the source, so the quality of the total fit alone is not enough: inspect how the model allocates
light between the components.

__Source Light__

An MGE can also describe a lensed source. Solving its intensities and fixing a suitable range of Gaussian
widths can reduce the number of parameters explored by the non-linear search.

Here the MGE describes the source before lensing. The lens mass model maps that light into the image plane,
where the predicted arcs are compared with the data. This provides a flexible smooth source model without
requiring a separate non-linear intensity parameter for every Gaussian.

The second model in `modeling.py` demonstrates an MGE source. The `source_science.py` example uses one to
calculate source flux and magnification, and explores how changing the assumed source light profile affects
those measurements. These quantities therefore depend on the source model as well as the lens mass model.

However, the same modelling restrictions apply. A common-centre MGE may be poorly suited to separated
star-forming clumps or multiple background galaxies. Changing the basis can add flexibility, but it also
changes the model you are asking the data to constrain.

If your aim is to reconstruct detailed, irregular source morphology, read the pixelization introduction as
well. Follow the available source examples below to see how an MGE source is constructed.

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

__Computational and Scientific Trade-offs__

Evaluating many Gaussian components and solving their intensities adds work to each likelihood calculation. On
the other hand, reducing the number of sampled parameters can make the non-linear search easier. The balance
depends on the dataset and model; an MGE is not guaranteed to make every fit faster.

An MGE is also less directly interpretable than a single Sersic profile. An individual Gaussian is usually a
component of the mathematical description, rather than a separate physical structure in the galaxy. Quantities
such as total luminosity or a half-light radius should be derived from the combined light distribution.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF. The
foreground light, deflecting mass and background source play different roles; retain the particular component
choices stated in each recipe.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/fit.ipynb) — Apply a fixed model and inspect the fit.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/modeling.ipynb) — Fit MGE lens light, then explore the MGE source example.
- [likelihood_function.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/likelihood_function.ipynb) — Follow the likelihood calculation.
- [simulator.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/simulator.ipynb) — Generate the example dataset.
- [slam.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/slam.ipynb) — Follow the staged SLaM fit.
- [source_science.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/source_science.ipynb) — Calculate MGE source flux and magnification and examine model dependence.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
