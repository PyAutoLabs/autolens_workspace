"""
Fits: Group
===========

This guide shows how to fit data using the `FitImaging` object for group-scale strong lenses, including visualizing
and interpreting its results.

A group-scale lens differs from a galaxy-scale lens in that there are multiple lens galaxies contributing to the
lensing. In this example, there is a single main lens galaxy and two extra galaxies nearby whose mass contributes
significantly to the ray-tracing and must therefore be included in the model.

References
----------

This example uses functionality described fully in other examples in the `guides` package:

- `guides/plot`: Using the plotting API (`aplt.plot_array`, `aplt.subplot_fit_imaging`, etc.) to visualize figures.
- `guides/units`: The source code unit conventions (e.g. arc seconds for distances and how to convert to physical units).
- `guides/data_structures`: The bespoke data structures used to store 1D and 2d arrays.

__Contents__

**Loading Data:** We begin by loading the group-scale strong lens dataset `simple` from .fits files, which is the.
**Mask:** Define the 2D mask applied to the dataset for the model-fit.
**Galaxy Centres:** For group-scale lenses we load the centres of the main lens galaxies and extra galaxies from JSON.
**Fitting:** Fit the lens model to the dataset and inspect the results.
**Bad Fit:** A bad lens model will show features in the residual-map and chi-squared map.
**Fit Quantities:** The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
**Figures of Merit:** There are single valued floats which quantify the goodness of fit.
**Plane Quantities:** The `FitImaging` object has specific quantities which break down each image of each plane.
**Unmasked Quantities:** All of the quantities above are computed using the mask which was used to fit the data.
**Pixel Counting:** An alternative way to quantify residuals like the lens light residuals is pixel counting.
**Outputting Results:** You may wish to output certain results to .fits files for later inspection.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Loading Data__

We begin by loading the group-scale strong lens dataset `simple` from .fits files, which is the dataset
we will use to demonstrate fitting.

This dataset was simulated using the `group/simulator` example, read through that to have a better
understanding of how the data this example fits was generated.

The group-scale dataset has a larger field of view than a typical galaxy-scale lens, because it includes
emission from multiple lens galaxies and a more extended lensing configuration.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

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
The `aplt.subplot_imaging_dataset` contains a subplot which plots all the key properties of the dataset simultaneously.

This includes the observed image data, RMS noise map, Point Spread Function and other information.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We now mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

We use a ``Mask2D`` object, which for this example is a 7.5" circular mask. This is larger than a typical
galaxy-scale lens mask because the group-scale lens has emission spread over a wider area due to the
multiple lens galaxies.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

"""
We now combine the imaging dataset with the mask.
"""
dataset = dataset.apply_mask(mask=mask)

"""
We now plot the image with the mask applied, where the image automatically zooms around the mask to make the lensed
source appear bigger.
"""
aplt.plot_array(array=dataset.data, title="Image Data With Mask Applied")

"""
The mask is also used to compute a `Grid2D`, where the (y,x) arc-second coordinates are only computed in unmasked
pixels within the masks' circle.

As shown in the previous overview example, this grid will be used to perform lensing calculations when fitting the
data below.
"""
aplt.plot_grid(grid=dataset.grid, title="Grid2D of Masked Dataset")

"""
__Galaxy Centres__

For group-scale lenses we load the centres of the main lens galaxies and extra galaxies from JSON files. These
centres are used during modeling to fix or constrain the positions of the galaxies.

The main lens galaxy is at (0.0, 0.0) and the two extra galaxies are at (3.5, 2.5) and (-4.4, -5.0).
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

print(f"Main lens centres: {main_lens_centres}")
print(f"Extra galaxies centres: {extra_galaxies_centres}")

"""
__Fitting__

Following the previous overview example, we can make a tracer from a collection of light profiles, mass profiles
and galaxies.

The combination of light and mass profiles below is the same as those used to generate the simulated
dataset we loaded above.

For a group-scale lens, we have multiple lens galaxies: a main lens galaxy and extra galaxies. The fit
handles all of these galaxies simultaneously, computing the combined deflection field from all mass
profiles to ray-trace the source galaxy light.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

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

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
Because the tracer's light and mass profiles are the same used to make the dataset, its image is nearly the same as the
observed image.

However, the tracer's image does appear different to the data, in that its ring appears a bit thinner. This is
because its image has not been blurred with the telescope optics PSF, which the data has.

[For those not familiar with Astronomy data, the PSF describes how the observed emission of the galaxy is blurred by
the telescope optics when it is observed. It mimicks this blurring effect via a 2D convolution operation].
"""
aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer  Image")

"""
We now use a `FitImaging` object to fit this tracer to the dataset.

The fit creates a `model_image` which we fit the data with, which includes performing the step of blurring the tracer`s
image with the imaging dataset's PSF. We can see this by comparing the tracer`s image (which isn't PSF convolved) and
the fit`s model image (which is).

For a group-scale lens, the model image includes contributions from all lens galaxies (main and extra) as well as
the lensed source galaxy.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.plot_array(array=fit.model_data, title="Model Image")

"""
The fit does a lot more than just blur the tracer's image with the PSF, it also creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset`s `data`.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:
"""
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi Squared Map")

"""
A subplot can be plotted which contains all of the above quantities, as well as other information contained in the
tracer such as the source-plane image, a zoom in of the source-plane and a normalized residual map where the colorbar
goes from 1.0 sigma to -1.0 sigma, to highlight regions where the fit is poor.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
The fit also provides us with a ``log_likelihood``, a single value quantifying how good the tracer fitted the dataset.

Lens modeling, described in the next overview example, effectively tries to maximize this log likelihood value.
"""
print(fit.log_likelihood)

"""
__Bad Fit__

A bad lens model will show features in the residual-map and chi-squared map.

We can produce such an image by creating a tracer with different lens and source galaxies. In the example below, we
change the centre of the main lens galaxy's mass from (0.0, 0.0) to (0.2, 0.2), which leads to residuals appearing
in the fit. For a group-scale lens, even a small offset in the main lens mass centre can produce significant
residuals because the main lens dominates the total deflection field.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.2, 0.2), einstein_radius=4.0),
)

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

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

"""
A new fit using this tracer shows residuals, normalized residuals and chi-squared which are non-zero.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
We also note that its likelihood decreases.
"""
print(fit.log_likelihood)

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

There is a `model_image`, which is the image-plane image of the tracer we inspected in the previous tutorial
blurred with the imaging data's PSF.

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_data.slim)

# The native property provides quantities in 2D NumPy Arrays.
# print(fit.model_data.native)

"""
There are numerous ndarrays showing the goodness of fit:

 - `residual_map`: Residuals = (Data - Model_Data).
 - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
 - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
"""
print(fit.residual_map.slim)
print(fit.normalized_residual_map.slim)
print(fit.chi_squared_map.slim)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.

 - `noise_normalization`: The normalizing noise term in the likelihood function
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).

 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Plane Quantities__

The `FitImaging` object has specific quantities which break down each image of each plane:

 - `model_images_of_planes_list`: Model-images of each individual plane, which for a group-scale lens includes the
 model images of the main lens galaxy, each extra galaxy and the lensed source galaxy. All images are convolved
 with the imaging's PSF.

 - `subtracted_images_of_planes_list`: Subtracted images of each individual plane, which are the data's image with
   all other plane's model-images subtracted. For example, the first subtracted image has the source galaxy's and
   extra galaxies' model images subtracted, leaving only the main lens galaxy's emission. This is especially useful
   for group-scale lenses where isolating the light contribution of each galaxy is important.

For group-scale lenses, there are more galaxies contributing to each plane compared to galaxy-scale lenses.
All lens galaxies (main and extra) are at the same redshift and therefore in the same plane, while the
source galaxy is in a separate background plane.
"""
print(fit.model_images_of_planes_list[0].slim)
print(fit.model_images_of_planes_list[1].slim)

print(fit.subtracted_images_of_planes_list[0].slim)
print(fit.subtracted_images_of_planes_list[1].slim)

"""
__Unmasked Quantities__

All of the quantities above are computed using the mask which was used to fit the data.

The `FitImaging` can also compute the unmasked blurred image of each plane.
"""
print(fit.unmasked_blurred_image.native)
print(fit.unmasked_blurred_image_of_planes_list[0].native)
print(fit.unmasked_blurred_image_of_planes_list[1].native)

"""
__Mask__

We can use the `Mask2D` object to mask regions of one of the fit's maps and estimate quantities of it.

Below, we estimate the average absolute normalized residuals within a 1.0" circular mask, which would inform us of
how accurate the lens light subtraction of a model fit is and if it leaves any significant residuals.

For group-scale lenses, this is particularly useful for evaluating how well each individual galaxy's light
has been subtracted.
"""
mask = al.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

normalized_residuals = fit.normalized_residual_map.apply_mask(mask=mask)

print(np.mean(np.abs(normalized_residuals.slim)))

"""
__Pixel Counting__

An alternative way to quantify residuals like the lens light residuals is pixel counting. For example, we could sum
up the number of pixels whose chi-squared values are above 10 which indicates a poor fit to the data.

Whereas computing the mean above the average level of residuals, pixel counting informs us how spatially large the
residuals extend.
"""
mask = al.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

chi_squared_map = fit.chi_squared_map.apply_mask(mask=mask)

print(np.sum(chi_squared_map > 10.0))

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection.

For example, one could output the lens light subtracted image of the lensed source galaxy to a .fits file such that
we could fit this source-only image again with an independent pipeline. For group-scale lenses, this subtracted
image has the light of all lens galaxies (main and extra) removed.
"""
lens_subtracted_image = fit.subtracted_images_of_planes_list[1]
aplt.fits_array(
    array=lens_subtracted_image,
    file_path=dataset_path / "lens_subtracted_data.fits", overwrite=True
)

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

For group-scale lenses, we apply adaptive over-sampling at the centres of all galaxies in the group, including
both the main lens galaxies and the extra galaxies. This ensures that the light profiles of every galaxy
in the group are accurately evaluated.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
Now we re-create the correct tracer and perform the fit with over-sampling applied.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

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

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(fit.log_likelihood)

"""
Fin.
"""
