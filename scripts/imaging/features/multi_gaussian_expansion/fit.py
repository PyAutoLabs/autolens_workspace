"""
Modeling Features: Multi Gaussian Expansion
===========================================

A multi Gaussian expansion (MGE) decomposes the lens light into ~15-100 Gaussians, where the `intensity` of every
Gaussian is solved for via a linear algebra using a process called an "inversion" (see the `light_parametric_linear.py`
feature for a full description of this).

This script fits a lens light model which uses an MGE consisting of 60 Gaussians. It is fitted to simulated data
where the lens galaxy's light has asymmetric and irregular features, which fitted poorly by symmetric light
profiles like the `Sersic`.

__Contents__

**Advantages & Disadvantages:** Benefits and drawbacks of using an MGE.
**Positive Only Solver:** How a positive solution to the light profile intensities is ensured.
**MGE Source Galaxy:** Discussion of using the MGE for the source galaxy, which is illustrated fully at the end of the example.
**Dataset & Mask:** Standard set up of imaging dataset that is fitted.
**Basis:** How to create a basis of multiple light profiles, in this example Gaussians.
**Gaussians:** A visualization of the Gaussians in the Basis that make up the MGE.
**Linear Light Profiles:** How to create a basis of linear light profiles to perform the MGE.
**Fit:** Perform a fit to a dataset using linear light profile MGE.
**Intensities:** Access the solved for intensities of linear light profiles from the fit.

__Advantages__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
An MGE fully captures these features and can therefore much better represent the emission of complex lens galaxies.

The MGE model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this example,
two separate groups of Gaussians are used to represent the `bulge` and `disk` of the lens, which in total correspond
to just N=6 non-linear parameters (a `bulge` and `disk` comprising two linear Sersics has N=10 parameters).

The MGE model parameterization is also composed such that neither the `intensity` parameters or any of the
parameters controlling the size of the Gaussians (their `sigma` values) are non-linear parameters sampled by Nautilus.
This removes the most significant degeneracies in parameter space, making the model much more reliable and efficient
to fit.

Therefore, not only does an MGE fit more complex galaxy morphologies, it does so using fewer non-linear parameters
in a much simpler non-linear parameter space which has far less significant parameter degeneracies!

__Disadvantages__

To fit an MGE model to the data, the light of the ~15-75 or more Gaussian in the MGE must be evaluated and compared
to the data. This is slower than evaluating the light of ~2-3 Sersic profiles, producing slower computational run
times (although the simpler non-linear parameter space will speed up the fit overall).

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical. For an MGE, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoLens** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__MGE Source Galaxy__

The MGE was designed to model the light of lens galaxies, because they are typically elliptical galaxies whose
morphology is better represented as many Gaussians. Their complex features (e.g. isophotal twists,
an ellipticity which varies radially) are accurately captured by an MGE.

The morphological features typically seen in source galaxies (e.g. disks, bars, clumps of star formation) are less
suited to an MGE. The source-plane of many lenses also often have multiple galaxies, whereas the MGE fitted
in this example assumes a single `centre`.

However, despite these limitations, an MGE turns out to be an extremely powerful way to model the source galaxies
of strong lenses. This is because, even if it struggles to capture the source's morphology, the simplification of
non-linear parameter space and removal of degeneracies makes it much easier to obtain a reliable lens model.
This is driven by the removal of any non-linear parameters which change the size of the source's light profile,
which are otherwise the most degenerate with the lens's mass model.

The second example in this script therefore uses an MGE source. We strongly recommend you read that example and adopt
MGE lens light models and source models, instead of the elliptical Sersic profiles, as soon as possible!

To capture the irregular and asymmetric features of the source's morphology, or reconstruct multiple source galaxies,
we recommend using a pixelized source reconstruction (see `autolens_workspace/modeling/features/pixelization.py`).
Combining this with an MGE for the len's light can be a very powerful way to model strong lenses!

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's bulge is a super position of 60 `Gaussian`` profiles.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and plot the strong lens dataset `simple` via .fits files.
"""
dataset_name = "lens_light_asymmetric"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Apply adaptive over sampling to ensure the lens galaxy light calculation is accurate, you can read up on over-sampling 
in more detail via the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Basis__

We first build a `Basis`, which is built from multiple light profiles (in this case, Gaussians). 

Below, we make a `Basis` out of 30 elliptical Gaussian light profiles which: 

 - All share the same centre and elliptical components.
 - The `sigma` size of the Gaussians increases in log10 increments.

Note that any light profile can be used to compose a Basis, but Gaussians are a good choice for lens galaxies
because they can capture the structure of elliptical galaxies.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
__Gaussians__

The `Basis` is composed of many Gaussians, each with different sizes (the `sigma` value) and therefore capturing
emission on different scales.

These Gaussians are visualized below using a `BasisPlotter`, which shows that the Gaussians expand in size as the
sigma value increases, in log10 increments.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

basis_plotter = aplt.BasisPlotter(basis=bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Linear Light Profiles__

We now show Composing a basis of multiple Gaussians and use them to fit the lens galaxy's light in data.

This does not perform a model-fit via a non-linear search, and therefore requires us to manually specify and guess
suitable parameter values for the shapelets (e.g. the `centre`, `ell_comps`). However, Gaussians are
very flexible and will give us a decent looking lens fit even if we just guess sensible values
for each parameter. 

The one parameter that is tricky to guess is the `intensity` of each Gaussian. A wide range of positive `intensity` 
values are required to decompose the lens galaxy's light accurately. We certainly cannot obtain a good solution by 
guessing the `intensity` values by eye.

We therefore use linear light profile Gaussians, which determine the optimal value for each Gaussian's `intensity` 
via linear algebra. Linear light profiles are described in the `linear_light_profiles.py` example and you should
familiarize yourself with this example before using the multi-Gaussian expansion.

We therefore again setup a `Basis` in an analogous fashion to the previous example, but this time we use linear
Gaussians (via the `lp_linear.linear` module).
"""
total_gaussians = 60

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".

mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# A list of linear light profile Gaussians will be input here, which will then be used to fit the data.

bulge_gaussian_list = []

# Iterate over every Gaussian and create it, with it centered at (0.0", 0.0") and assuming spherical symmetry.

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

# The Basis object groups many light profiles together into a single model component and is used to fit the data.

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

"""
__Fit__

This is to illustrate the API for performing an MGE using standard autolens objects like the `Galaxy`, `Tracer`
and `FitImaging` 

Once we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Tracer` and 
use it to fit data.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=bulge,
)

tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
By plotting the fit, we see that the MGE does a reasonable job at capturing the appearance of the lens galaxy.

The majority of residuals are due to the lensed source, which was not included in the model. There are faint
central residuals, which are due to the MGE not being a perfect fit to the lens galaxy's light. 

Given that there was no non-linear search to determine the optimal values of the Gaussians and the source galaxy
was omitted entirely, this is a pretty good fit!
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
We can use the `BasisPlotter` to plot each individual Gaussian in the reconstructed basis.

This plot shows each Gaussian has a unique positive `intensity` that was solved for via linear algebra.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

basis_plotter = aplt.BasisPlotter(basis=tracer.galaxies[0].bulge, grid=grid)
basis_plotter.subplot_image()

"""
__Intensities__

The fit contains the solved for intensity values.

These are computed using a fit's `linear_light_profile_intensity_dict`, which maps each linear light profile 
in the model parameterization above to its `intensity`.

The code below shows how to use this dictionary, as an alternative to using the max_log_likelihood quantities above.
"""
lens_bulge = fit.tracer.galaxies[0].bulge

print(
    f"\n Intensity of lens galaxy's first Gaussian in bulge = {fit.linear_light_profile_intensity_dict[lens_bulge.profile_list[0]]}"
)

"""
A `Tracer` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible from a fit.

For example, the first linear light profile of the MGE `bulge` component above printed it solved for intensity value,
but it was still represented as a linear light profile. 

The `tracer` created below instead has a standard light profile with an `intensity` actually set.

The benefit of using a tracer with standard light profiles is it can be visualized, as performed above (linear 
light profiles cannot by default because they do not have `intensity` values).
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

print(tracer.galaxies[0].bulge.profile_list[0].intensity)

"""
__Wrap Up__

A Multi Gaussian Expansion is a powerful tool for modeling the light of galaxies, and offers a compelling method to
fit complex light profiles with a small number of parameters.

Now you are familiar with MGE modeling, it is recommended you adopt this as your default lens modeling approach. 
However, it may not be suitable for lower resolution data, where the simpler Sersic profiles may be more appropriate.
"""
