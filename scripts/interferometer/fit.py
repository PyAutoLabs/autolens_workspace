"""
Fits
====

This guide shows how to fit data using the `FitInterferometer` object, including visualizing and interpreting its results.

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
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Loading Data__

We we begin by loading the strong lens dataset `simple` from .fits files, which is the dataset 
we will use to demonstrate fitting.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.

This dataset was simulated using the `interferometer/simulator` example, read through that to have a better
understanding of how the data this exam fits was generated. The simulation uses the `TransformerDFT` to map
the real-space image to the uv-plane.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
The `InterferometerPlotter` contains a subplot which plots all the key properties of the dataset simultaneously.

This includes the observed visibility data, RMS noise map and other information.
"""
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
Visibility data is in uv space, making it hard to interpret by eye.

The dirty images of the interferometer dataset can plotted, which use the transformer of the interferometer 
to map the visibilities, noise-map or other quantity to a real-space image.
"""
dataset_plotter.subplot_dirty_images()

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
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Because the tracer's light and mass profiles are the same used to make the dataset, its image is nearly the same as the
observed image.

We can plot the image of the tracer to confirm this, noting that for a tracer its images are always in real space
(not Fourier space like the interferometer dataset) and therefore they can be directly visualized.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.set_title("Tracer  Image")
tracer_plotter.figures_2d(image=True)

"""
However, the tracer's image is not what we observe in the interferometer dataset, because we observe the image as
visibilities in the uv-plane. 

To compare directly to the data, we therefore need to Fourier transform the tracer's image to the uv-plane. 

We do this by creating a `FitInterferometer` object, which performs this Fourier transform as part of the fitting 
procedure.

The code plots the result of this, by using the `model_data` of the fit, which performs this Fourier transform 
on the tracer image above and plots the result visibilities in uv-space.
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d(model_data=True)

"""
The visibilities are again hard to interpret by eye, so we can plot the dirty image of the fit's model data. This 
dirty image is the Fourier transform of the fit's model data (therefore the Fourier transform of the tracer's image) and
can be compared directly to the image of the tracer above (albeit it still has the interferometer's PSF/dirty beam
convolved with it).
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.figures_2d(dirty_image=True)

"""
The fit does a lot more than just Fourier transform the tracer's image it also creates the following:

 - The `residual_map`: The `model_data` visibilities subtracted from the observed dataset`s `data` visibilities.
 - The `normalized_residual_map`: The `residual_map `divided by the observed dataset's `noise_map`.
 - The `chi_squared_map`: The `normalized_residual_map` squared.

For a good lens model where the model and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:
"""
fit_plotter.figures_2d(
    residual_map_real=True,
    residual_map_imag=True,
    normalized_residual_map_real=True,
    normalized_residual_map_imag=True,
    chi_squared_map_real=True,
    chi_squared_map_imag=True,
)

"""
A subplot can be plotted which contains all of the above quantities, as well as other information contained in the
tracer such as the source-plane image, a zoom in of the source-plane and a normalized residual map where the colorbar
goes from 1.0 sigma to -1.0 sigma, to highlight regions where the fit is poor.
"""
fit_plotter.subplot_fit()

"""
Once again, dirty images are often easier to interpret, so we can plot a subplot of the dirty images of the data, model
data, residuals and chi-squared.
"""
fit_plotter.subplot_fit_dirty_images()

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
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
We also note that its likelihood decreases.
"""
print(fit.log_likelihood)

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

There is a `model_data`, which is the image-plane visibilities of the tracer.

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the 
goodness-of-fit.

If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(fit.model_data)

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
There are `dirty` variants of the above maps, which transform the visibilities, residual-map, chi squared and other
values to to real-space images using the interferometer's transformer.

These real space images can be mapped between their `slim` and `native` representations (see the
`guides/data_structures` example for more information on these terms).
"""
print(fit.dirty_image.slim)  # Data
print(fit.dirty_model_image.slim)
print(fit.dirty_residual_map.slim)
print(fit.dirty_normalized_residual_map.slim)
print(fit.dirty_chi_squared_map.slim)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.

 - `noise_normalization`: The normalizing noise term in the likelihood function 
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).

 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
 
These sum other both the real and imaginary components of the visibilities to give a single value for each quantity.
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)

"""
__Plane Quantities__

The `FitInterferometer` object has specific quantities which break down each image of each plane:

 - `model_visibilities_of_planes_list`: Model-images of each individual plane, which in this example is a model image of the 
 lens galaxy and model image of the lensed source galaxy, both corresponding to dirty images.

 - `subtracted_images_of_planes_list`: Subtracted images of each individual plane, which are the data's image with
   all other plane's model-images subtracted. For example, the first subtracted image has the source galaxy's model image
   subtracted and therefore is of only the lens galaxy's emission. The second subtracted image is of the lensed source,
   with the lens galaxy's light removed.

For multi-plane lens systems these lists will be extended to provide information on every individual plane.
"""
print(fit.model_visibilities_of_planes_list[1].slim)

"""
There is also a `galaxy_model_visibilities_dict` which maps each galaxy in the tracer to its model visibilities.
"""
print(fit.galaxy_model_visibilities_dict[source_galaxy].slim)

"""
A dictionary which maps the model images of each galaxy is also available.

These are not the dirty images, but instead the images of each galaxy that come from the tracer object
(e.g. simply evaluating the tracer's image on the interferometer's real-space grid).
"""
print(fit.galaxy_model_image_dict[source_galaxy].slim)

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection. 

For example, one could output the lens light subtracted image of the lensed source galaxy to a .fits file such that
we could fit this source-only image again with an independent pipeline.
"""
source_model_image = fit.galaxy_model_image_dict[source_galaxy]
source_model_image.output_to_fits(
    file_path=dataset_path / "source_model_image.fits", overwrite=True
)

"""
Fin.
"""
