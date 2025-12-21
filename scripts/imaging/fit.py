"""
Fits
====

This guide shows how to fit data using the `FitImaging` object, including visualizing and interpreting its results.

References
----------

This example uses functionality described fully in other examples in the `guides` package:

- `guides/plot`: Using Plotter objects to plot and customize figures.
- `guides/units`: The source code unit conventions (e.g. arc seconds for distances and how to convert to physical units).
- `guides/data_structures`: The bespoke data structures used to store 1D and 2d arrays.
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

We we begin by loading the strong lens dataset `simple__no_lens_light` from .fits files, which is the dataset 
we will use to demonstrate fitting.

This dataset was simulated using the `imaging/simulator` example, read through that to have a better
understanding of how the data this exam fits was generated.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
The `ImagingPlotter` contains a subplot which plots all the key properties of the dataset simultaneously.

This includes the observed image data, RMS noise map, Point Spread Function and other information.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We now mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

We use a ``Mask2D`` object, which for this example is a 3.0" circular mask.
"""
mask_radius = 3.0

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
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.set_title("Image Data With Mask Applied")
dataset_plotter.figures_2d(data=True)

"""
The mask is also used to compute a `Grid2D`, where the (y,x) arc-second coordinates are only computed in unmasked 
pixels within the masks' circle. 

As shown in the previous overview example, this grid will be used to perform lensing calculations when fitting the
data below.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.grid)
grid_plotter.set_title("Grid2D of Masked Dataset")
grid_plotter.figure_2d()

"""
__Fitting__

Following the previous overview example, we can make a tracer from a collection of light profiles, mass profiles
and galaxies.

The combination of light and mass profiels below is the same as those used to generate the simulated 
dataset we loaded above.

It therefore produces a tracer whose image looks exactly like the dataset.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Because the tracer's light and mass profiles are the same used to make the dataset, its image is nearly the same as the
observed image.

However, the tracer's image does appear different to the data, in that its ring appears a bit thinner. This is
because its image has not been blurred with the telescope optics PSF, which the data has.

[For those not familiar with Astronomy data, the PSF describes how the observed emission of the galaxy is blurred by
the telescope optics when it is observed. It mimicks this blurring effect via a 2D convolution operation].
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.set_title("Tracer  Image")
tracer_plotter.figures_2d(image=True)

"""
We now use a `FitImaging` object to fit this tracer to the dataset. 

The fit creates a `model_image` which we fit the data with, which includes performing the step of blurring the tracer`s 
image with the imaging dataset's PSF. We can see this by comparing the tracer`s image (which isn't PSF convolved) and 
the fit`s model image (which is).
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.figures_2d(model_image=True)

"""
The fit does a lot more than just blur the tracer's image with the PSF, it also creates the following:

 - The `residual_map`: The `model_image` subtracted from the observed dataset`s `data`.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:
"""
fit_plotter.figures_2d(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)

"""
A subplot can be plotted which contains all of the above quantities, as well as other information contained in the
tracer such as the source-plane image, a zoom in of the source-plane and a normalized residual map where the colorbar
goes from 1.0 sigma to -1.0 sigma, to highlight regions where the fit is poor.
"""
fit_plotter.subplot_fit()

"""
The fit also provides us with a ``log_likelihood``, a single value quantifying how good the tracer fitted the dataset.

Lens modeling, describe in the next overview example, effectively tries to maximize this log likelihood value.
"""
print(fit.log_likelihood)

"""
__Bad Fit__

A bad lens model will show features in the residual-map and chi-squared map.

We can produce such an image by creating a tracer with different lens and source galaxies. In the example below, we 
change the centre of the source galaxy from (0.0, 0.0) to (0.05, 0.05), which leads to residuals appearing
in the fit.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.1, 0.1),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
A new fit using this plane shows residuals, normalized residuals and chi-squared which are non-zero. 
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

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

 - `model_images_of_planes_list`: Model-images of each individual plane, which in this example is a model image of the 
 lens galaxy and model image of the lensed source galaxy. Both images are convolved with the imaging's PSF.

 - `subtracted_images_of_planes_list`: Subtracted images of each individual plane, which are the data's image with
   all other plane's model-images subtracted. For example, the first subtracted image has the source galaxy's model image
   subtracted and therefore is of only the lens galaxy's emission. The second subtracted image is of the lensed source,
   with the lens galaxy's light removed.

For multi-plane lens systems these lists will be extended to provide information on every individual plane.
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
how accurate the lens light subtraction of a model fit is and if it leaves any significant residuals
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
we could fit this source-only image again with an independent pipeline.
"""
lens_subtracted_image = fit.subtracted_images_of_planes_list[1]
lens_subtracted_image.output_to_fits(
    file_path=dataset_path / "lens_subtracted_data.fits", overwrite=True
)

"""
Fin.
"""
