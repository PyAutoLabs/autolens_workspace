"""
Features: Shapelets
===================

Shapelets provide a flexible model of galaxy light when a single smooth profile cannot describe the
structure seen in the data. In strong lensing, a key use is source science: reconstructing the background
galaxy from its distorted, magnified images to investigate its resolved morphology.

Use this feature to model departures from a smooth elliptical source, explore foreground galaxy-light
models where appropriate, and test how the choice of light representation affects the inferred lens model.
Shapelets offer an alternative to smooth parametric profiles, Multi-Gaussian Expansions (MGEs) and pixelized
reconstructions; the useful choice depends on the structure and information present in the data.

[Birrer, Amara & Refregier (2015)](https://arxiv.org/abs/1504.07629) demonstrate strong-lens modelling with
basis sets, including an application to RX J1131-1231. This is an example of the scientific method, rather
than a PyAutoLens application. [Refregier (2003)](https://arxiv.org/abs/astro-ph/0105178) introduces the
shapelet formalism and its image-analysis applications.

![Illustrative shapelet building blocks and a structured source model](../../../../../images/features/shapelets_illustration.png)

The illustration conveys the idea of combining mathematical building blocks into a galaxy-light model.
It is not a fitted reconstruction or the exact sum of the four displayed patterns. The colours of the
basis patterns distinguish positive and negative values, not different physical components of a galaxy.

__Basis Functions and Their Sum__

Shapelets describe light as a weighted combination of localized basis functions. Different functions
represent different spatial patterns, allowing their sum to describe structures beyond a single smooth
light profile.

An individual basis function is a mathematical building block, not a separate star-forming region, disk or
other physical component. The combined light distribution is the quantity to compare with observations
and interpret scientifically.

The following grid shows individual polar shapelet functions used to illustrate the basis:

![Individual polar shapelet basis functions](../../../../../images/features/shapelets_basis.png)

The basis order controls which modes are included. The scale parameter, beta, sets their characteristic
size, while the centre locates the basis within the galaxy. The examples also discuss shared ellipticity
parameters and polar versus Cartesian representations. These choices determine which spatial structures a
finite basis can represent; neither representation guarantees an adequate model for every galaxy.

__Linear Coefficients and Non-Linear Geometry__

For a chosen basis geometry and lens model, the coefficient of each shapelet is solved using linear
algebra. The non-linear search varies the parameters left free in the model, such as shared geometry and
the lens mass parameters.

This allows many basis functions to share a relatively small set of non-linear parameters. It does not
mean that the model has only that many degrees of freedom: the linear coefficients also describe the light
distribution. The number of functions and free geometric parameters depends on the construction in the
particular example, so read those choices beside the model definition.

Adding functions increases flexibility and the work required by the linear solve. A smaller non-linear
parameter space therefore does not automatically mean a faster fit than a simple Sersic model.

__Signed Coefficients and Physical Light__

Shapelet basis functions can contain positive and negative regions, and the solver allows coefficients of
either sign. These signed contributions are part of the mathematical representation; a negative coefficient
does not by itself imply negative physical emission.

The physical concern is negative surface brightness in the summed galaxy model. A finite basis can produce
negative regions or oscillatory structure, especially when it cannot represent the true morphology well.
Inspect the combined reconstruction as well as the image-plane residuals.

The shapelet examples use a solver that permits signed coefficients. This differs from the positive-only
solutions used in the MGE and pixelization examples. Do not impose positive-only shapelet coefficients as a
substitute for checking whether the summed brightness distribution is physically sensible.

__Flexibility and Its Limits__

Shapelets can describe some asymmetries, changes in apparent shape and other departures from a smooth
elliptical profile. They can nevertheless struggle with sharp features, compact star-forming knots, bars
or widely separated components. A basis sharing one centre may be inefficient for a merger or a galaxy
with bright companions.

Increasing the maximum order is not evidence that newly reconstructed detail is constrained by the data.
Compare plausible basis orders and scales, inspect residuals and source reconstructions together, and
check that the model is describing signal rather than noise. The PSF, sampling and lensing magnification
all affect the detail that can be recovered.

Regularization can provide another way to control flexibility, but it is a modelling assumption rather
than a guarantee of physical structure. The modeling example discusses this option and points to a
developer example; do not assume that every shapelet fit shown here uses regularization.

__What the Examples Teach__

Start with fit.py to inspect the basis, solved coefficients and combined light distribution for a fixed
model. Then follow modeling.py to see how the basis is embedded in a lens fit, which parameters are shared
or varied, and how polar and Cartesian shapelets are constructed.

The main lensing example models the background source with shapelets. The script also demonstrates their
use for lens-galaxy light. Keep the basis definition, coefficient constraints and any regularization in view
when interpreting either application.

__In This Folder__

These examples fit imaging data, so the model is compared with image pixels after accounting for the PSF.
For a background source, its shapelet light distribution is first mapped through the lens model; foreground
lens light has a different role. Retain the particular component choices stated in each recipe, including
whether foreground light is present or omitted.

__Choose An Example__

- [fit.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/fit.ipynb) — Inspect a fixed model, its basis and the combined light distribution.
- [modeling.py](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/advanced/shapelets/modeling.ipynb) — Fit a lens and shapelet source, then explore alternative basis constructions and lens-light modelling.

The links open notebook versions. The corresponding Python scripts are in this folder; run them from the workspace root.
"""
