"""
Results: Fits
=============

This tutorial inspects the model's fit to the data using the  `FitImaging` object inferred by the non-linear 
search, for example visualizing and interpreting its results.

This includes inspecting the residuals, chi-squared and other goodness-of-fit quantities.

__Plot Module__

This example uses the **PyAutoLens** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoLens** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/imaging/results/examples/data_structure.ipynb` example.

__Other Models__

This tutorial does not use a pixelized source reconstruction or linear light profiles, which have their own dediciated
functionality that interfacts with the `FitImaging` object.

These are described in the dedicated example scripts `results/examples/linear.py` and `results/examples/pixelizaiton.py`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using Nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy, redshift=0.5, bulge=al.lp.Sersic, mass=al.mp.Isothermal
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Fit__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit` which we can visualize.
"""
fit = result.max_log_likelihood_fit

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

There is a `model_image`, which is the image-plane image of the tracer we inspected in the previous tutorial
blurred with the imaging data's PSF. 

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the 
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_image.slim)

# The native property provides quantities in 2D NumPy Arrays.
# print(fit.model_image.native)

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
lens_subtracted_image_2d = fit.subtracted_images_of_planes_list[1]
lens_subtracted_image_2d.output_to_fits(
    file_path=path.join(dataset_path, "lens_subtracted_data.fits"), overwrite=True
)

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by Nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

tracer = al.Tracer(galaxies=instance.galaxies)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Wrap Up__

In this tutorial, we saw how to inspect the quality of a model fit using the fit imaging object.

If you are modeling strong lenses using interferometer data or a point-source dataset, we cover the
corresponding fit objects in tutorials 6 and 7.
"""
