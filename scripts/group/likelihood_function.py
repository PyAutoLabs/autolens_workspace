"""
__Log Likelihood Function: Group__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data of
a group-scale strong lens, where there are multiple lens galaxies whose mass profiles all contribute to the
ray-tracing of the source galaxy.

This script has the following aims:

 - To provide a resource that authors can include in papers using, so that readers can understand the likelihood
 function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To illustrate how group-scale lensing differs from galaxy-scale lensing: multiple mass profiles from
 multiple galaxies all contribute to the deflection angles, meaning the total deflection is the sum of
 deflections from every galaxy in the group.

Accompanying this script is the imaging `likelihood_function.py` which provides the same step-by-step guide
for a single lens galaxy. This script extends that to the group scale.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Masked Image Grid:** To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.
**Main Lens Galaxy:** The main lens galaxy is at the centre of the group.
**Extra Galaxies:** The two extra galaxies are companion galaxies near the main lens.
**Source Galaxy Light Profile:** The source galaxy is fitted using an analytic light profile, in this example a cored elliptical.
**Lens Light:** Compute a 2D image of each lens galaxy's light and sum them together.
**Lens Galaxy Mass:** We next consider the mass profiles of all galaxies in the group.
**Ray Tracing:** To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\\theta$ from the.
**Source Image:** We pass the traced grid and blurring grid of coordinates to the source galaxy to evaluate its 2D.
**Convolution:** Convolve the 2D image of the lens galaxies and source above with the PSF in real-space (as opposed.
**Likelihood Function:** We now quantify the goodness-of-fit of our group-scale lens model.
**Chi Squared:** The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is.
**Noise Normalization Term:** Our likelihood function assumes the imaging data consists of independent Gaussian noise in every.
**Calculate The Log Likelihood:** We can now, finally, compute the `log_likelihood` of the lens model, by combining the two terms.
**Fit:** Fit the lens model to the dataset.
**Lens Modeling:** To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using.
**Wrap Up:** Summary of the script and next steps.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autoarray as aa
import autolens.plot as aplt

"""
__Dataset__

In order to perform a likelihood evaluation, we first load a dataset.

This example fits a simulated group-scale strong lens where the imaging resolution is 0.1 arcsecond-per-pixel
resolution. The group consists of one main lens galaxy and two extra companion galaxies whose mass contributes
significantly to the ray-tracing.
"""
dataset_path = Path("dataset", "group", "simple")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
This guide uses in-built visualization tools for plotting.

For example, using the `aplt.subplot_imaging_dataset` the imaging dataset we perform a likelihood evaluation on is plotted.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

The likelihood is only evaluated using image pixels contained within a 2D mask, which we choose before performing
lens modeling.

Below, we define a 2D circular mask with a 7.5" radius. This is larger than the mask used for galaxy-scale lenses
because group-scale systems have lensed images that extend over a wider area due to the combined mass of multiple
galaxies.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
When we plot the masked imaging, only the circular masked region is shown.
"""
aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Over Sampling__

Over sampling evaluates a light profile using multiple samples of its intensity per image-pixel.

For simplicity, we disable over sampling in this guide by setting `sub_size=1`.

a full description of over sampling and how to use it is given in `autolens_workspace/*/guides/over_sampling.py`.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Masked Image Grid__

To perform galaxy calculations we define a 2D image-plane grid of (y,x) coordinates.

These are given by `masked_dataset.grids.lp`, which we can plot and see is a uniform grid of (y,x) Cartesian
coordinates which have had the 7.5" circular mask applied.

Each (y,x) coordinate coordinates to the centre of each image-pixel in the dataset, meaning that when this grid is
used to perform ray-tracing and evaluate a light profile the intensity of the profile at the centre of each
image-pixel is computed, making it straight forward to compute the light profile's image to the image data.
"""
aplt.plot_grid(grid=masked_dataset.grids.lp, title="")

print(
    f"(y,x) coordinates of first ten unmasked image-pixels {masked_dataset.grid[0:9]}"
)

"""
__Lens Galaxy Light (Setup)__

To perform a likelihood evaluation we now compose our lens model.

For a group-scale lens, there are multiple galaxies whose light and mass must be modeled. We define each galaxy
individually.

A light profile is defined by its intensity $I (\eta_{\rm l}) $, for example the Sersic profile:

$I_{\rm  Ser} (\eta_{\rm l}) = I \exp \bigg\{ -k \bigg[ \bigg( \frac{\eta}{R} \bigg)^{\frac{1}{n}} - 1 \bigg] \bigg\}$

Where:

 - $\eta$ are the elliptical coordinates of the masked image-grid.
 - $I$ is the `intensity`, which controls the overall brightness of the Sersic profile.
 - $n$ is the ``sersic_index``, which via $k$ controls the steepness of the inner profile.
 - $R$ is the `effective_radius`, which defines the arc-second radius of a circle containing half the light.

__Main Lens Galaxy__

The main lens galaxy is at the centre of the group. It has a spherical Sersic light profile and a spherical
isothermal mass profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

"""
__Extra Galaxies__

The two extra galaxies are companion galaxies near the main lens. They each have their own light and mass profiles.

Getting the mass of these extra galaxies right is crucial: their mass profiles contribute to the total deflection
angles, and if they are wrong the ray-traced source-plane coordinates will be incorrect, leading to a poor fit.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

"""
__Source Galaxy Light Profile__

The source galaxy is fitted using an analytic light profile, in this example a cored elliptical Sersic.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
Using the masked 2D grid defined above, we can calculate and plot images of each galaxy's light profile.
"""
image_2d_lens = lens_galaxy.image_2d_from(grid=masked_dataset.grid)

aplt.plot_array(array=image_2d_lens, title="Main Lens Galaxy Image")

image_2d_extra_0 = extra_galaxy_0.image_2d_from(grid=masked_dataset.grid)

aplt.plot_array(array=image_2d_extra_0, title="Extra Galaxy 0 Image")

image_2d_extra_1 = extra_galaxy_1.image_2d_from(grid=masked_dataset.grid)

aplt.plot_array(array=image_2d_extra_1, title="Extra Galaxy 1 Image")

"""
__Lens Light__

Compute a 2D image of each lens galaxy's light and sum them together.

For group-scale lenses, the total lens light is the sum of the images of ALL lens galaxies (main + extra). This is
a key difference from galaxy-scale lensing where there is typically only one lens galaxy.
"""
lens_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grid)
extra_0_image_2d = extra_galaxy_0.image_2d_from(grid=masked_dataset.grid)
extra_1_image_2d = extra_galaxy_1.image_2d_from(grid=masked_dataset.grid)

total_lens_image_2d = lens_image_2d + extra_0_image_2d + extra_1_image_2d

aplt.plot_array(array=total_lens_image_2d, title="Total Lens Light (All Galaxies)")

"""
To convolve the lens light with the imaging data's PSF, we need the `blurring_image`. This represents all flux
values not within the mask, which are close enough to it that their flux blurs into the mask after PSF convolution.

We compute blurring images for ALL lens galaxies and sum them.
"""
lens_blurring_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)
extra_0_blurring_image_2d = extra_galaxy_0.image_2d_from(
    grid=masked_dataset.grids.blurring
)
extra_1_blurring_image_2d = extra_galaxy_1.image_2d_from(
    grid=masked_dataset.grids.blurring
)

total_lens_blurring_image_2d = (
    lens_blurring_image_2d + extra_0_blurring_image_2d + extra_1_blurring_image_2d
)

"""
__Lens Galaxy Mass__

We next consider the mass profiles of all galaxies in the group.

A mass profile is defined by its convergence $\kappa (\eta)$, which is related to
the surface density of the mass distribution as

$\kappa(\eta)=\frac{\Sigma(\eta)}{\Sigma_\mathrm{crit}},$

where

$\Sigma_\mathrm{crit}=\frac{{\rm c}^2}{4{\rm \pi} {\rm G}}\frac{D_{\rm s}}{D_{\rm l} D_{\rm ls}},$

For the isothermal profile used by all galaxies in this group:

$\kappa(\eta) = \frac{1.0}{1 + q} \bigg( \frac{\theta_{\rm E}}{\eta} \bigg)$

Where $\theta_{\rm E}$ is the `einstein_radius`.

From each mass profile we can compute its deflection angles, which describe how due to gravitational lensing
image-pixels are ray-traced to the source plane:

$\\vec{{\\alpha}}_{\\rm x,y} (\\vec{x}) = \\frac{1}{\\pi} \\int \\frac{\\vec{x} - \\vec{x'}}{\\left | \\vec{x} - \\vec{x'} \\right |^2} \\kappa(\\vec{x'}) d\\vec{x'} \\, ,$
"""
deflections_lens = lens_galaxy.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_0 = extra_galaxy_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_1 = extra_galaxy_1.deflections_yx_2d_from(grid=masked_dataset.grid)

"""
__Ray Tracing__

To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\\theta$ from the image-plane to its (y,x)
source-plane coordinate $\\beta$ using the summed deflection angles $\\alpha$ of ALL mass profiles:

 $\\beta = \\theta - \\alpha(\\theta)$

For group-scale lensing, the total deflection angle $\\alpha$ is the sum of deflection angles from ALL galaxies:

 $\\alpha_{\\rm total} = \\alpha_{\\rm lens} + \\alpha_{\\rm extra\\_0} + \\alpha_{\\rm extra\\_1}$

This is the fundamental reason why getting the mass of extra galaxies right matters: each galaxy's mass profile
contributes to the total deflection, and errors in any of them lead to incorrect source-plane coordinates.

The `Tracer` object handles this automatically by including all galaxies when computing ray-traced coordinates.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

# A list of every grid (e.g. image-plane, source-plane) however we only need the source plane grid with index -1.
traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

traced_blurring_grid = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)[-1]

aplt.plot_grid(grid=traced_blurring_grid, title="Source Plane Blurring Grid (Traced)")

"""
__Source Image__

We pass the traced grid and blurring grid of coordinates to the source galaxy to evaluate its 2D image.

This step is identical to galaxy-scale lensing -- the source galaxy's light profile is evaluated on the
source-plane grid. The difference is that the source-plane grid was computed using deflection angles from
multiple galaxies.
"""
source_image_2d = source_galaxy.image_2d_from(grid=traced_grid)

source_blurring_image_2d = source_galaxy.image_2d_from(grid=traced_blurring_grid)

"""
__Lens + Source Light Addition__

We add the total lens galaxy light (from ALL galaxies) and the source galaxy image together, to create an overall
image of the group-scale strong lens.
"""
image = total_lens_image_2d + source_image_2d

aplt.plot_array(array=image, title="Total Image (All Galaxies + Source)")

blurring_image_2d = total_lens_blurring_image_2d + source_blurring_image_2d

"""
__Convolution__

Convolve the 2D image of the lens galaxies and source above with the PSF in real-space (as opposed to via an FFT)
using a `Kernal2D`.
"""
convolved_image_2d = masked_dataset.psf.convolved_image_from(
    image=image, blurring_image=blurring_image_2d
)

aplt.plot_array(array=convolved_image_2d, title="Convolved Image")

"""
__Likelihood Function__

We now quantify the goodness-of-fit of our group-scale lens model.

We compute the `log_likelihood` of the fit, which is the value returned by the `log_likelihood_function`.

The likelihood function for parametric lens modeling consists of two terms:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

We now explain what each of these terms mean.

__Chi Squared__

The first term is a $\chi^2$ statistic, which is defined above in our merit function as and is computed as follows:

 - `model_data` = `convolved_image_2d`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

The chi-squared therefore quantifies if our fit to the data is accurate or not.

High values of chi-squared indicate that there are many image pixels our model did not produce a good fit to the image
for, corresponding to a fit with a lower likelihood.
"""
model_image = convolved_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
The `chi_squared_map` indicates which regions of the image we did and did not fit accurately.
"""
chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

aplt.plot_array(array=chi_squared_map, title="Chi-Squared Map")

"""
__Noise Normalization Term__

Our likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel.

The final term in the likelihood function is therefore a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared.

Given the `noise_map` is fixed, this term does not change during the lens modeling process and has no impact on the
model we infer.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We can now, finally, compute the `log_likelihood` of the lens model, by combining the two terms computed above using
the likelihood function defined above.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__Fit__

This step-by-step process to perform a likelihood function evaluation is what is performed in the `FitImaging` object.

The `FitImaging` object handles all of the steps above automatically: it sums the light from all galaxies, computes
the total deflection angles from all mass profiles, ray-traces the grid to the source plane, evaluates the source
light, convolves with the PSF, and computes the log likelihood.

For group-scale lenses, the key advantage of the `FitImaging` and `Tracer` objects is that they automatically handle
the summation of light and mass contributions from an arbitrary number of galaxies.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)

aplt.subplot_fit_imaging(fit=fit)

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `Nautilus` (https://github.com/joshspeagle/Nautilus)
multiple MCMC and optimization algorithms are supported.

__Wrap Up__

We have presented a visual step-by-step guide to the group-scale parametric likelihood function.

The key differences from galaxy-scale lensing are:

 - Multiple lens galaxies (main + extra) each contribute light profiles whose images are summed together.
 - Multiple mass profiles from ALL galaxies contribute to the deflection angles, and the total deflection
   is the sum of deflections from every galaxy.
 - Getting the mass of extra galaxies right is important because their deflection angles affect the
   source-plane coordinates and therefore the quality of the source reconstruction.
 - The `FitImaging` and `Tracer` objects handle all of this automatically for an arbitrary number of galaxies.
"""
