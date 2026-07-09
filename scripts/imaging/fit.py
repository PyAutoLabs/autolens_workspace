"""
Fits
====

This guide shows how to fit data using the `FitImaging` object, including visualizing and interpreting its results.

References
----------

This example uses functionality described fully in other examples in the `guides` package:

- `guides/plot`: Using the plotting API (`aplt.plot_array`, `aplt.subplot_fit_imaging`, etc.) to visualize figures.
- `guides/units`: The source code unit conventions (e.g. arc seconds for distances and how to convert to physical units).
- `guides/data_structures`: The bespoke data structures used to store 1D and 2d arrays.

__Contents__

- **Loading Data:** We we begin by loading the strong lens dataset `simple__no_lens_light` from .fits files, which is.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Fitting:** Fit the lens model to the dataset and inspect the results.
- **Bad Fit:** A bad lens model will show features in the residual-map and chi-squared map.
- **Fit Quantities:** The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.
- **Figures of Merit:** There are single valued floats which quantify the goodness of fit.
- **Plane Quantities:** The `FitImaging` object has specific quantities which break down each image of each plane.
- **Unmasked Quantities:** All of the quantities above are computed using the mask which was used to fit the data.
- **Pixel Counting:** An alternative way to quantify residuals like the lens light residuals is pixel counting.
- **Outputting Results:** You may wish to output certain results to .fits files for later inspection.

__JAX__

This script constructs a `FitImaging` directly from a tracer and dataset
(no Analysis / no non-linear search). The fit itself is JAX-friendly:
any quantities it computes (`fit.model_image`, `fit.residual_map`,
`fit.log_likelihood`, etc.) work on either backend and return arrays
backed by `numpy.ndarray` on the default path or `jax.Array` if you
constructed the upstream objects with `xp=jnp`.

For the standard analysis-driven modeling path — where `AnalysisImaging`
auto-enables `use_jax=True` and the search driver handles the JIT
internally — see `start_here.py` / `modeling.py`. For the advanced path
where you wrap your own `@jax.jit` around `FitImaging` construction, see
`likelihood_function.py`'s `__JAX__` section and the `lens_calc.py` guide.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

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

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

# PSF convolution runs at the image resolution (sub size 1), which is the fastest
# option and accurate for well-sampled PSFs. Supplying a PSF at a multiple of the
# image resolution and raising this value improves blurring fidelity for
# undersampled PSFs (e.g. HST / Euclid VIS) at extra compute cost — see
# `guides/advanced/over_sampling.py` and the simulator's `__Oversampled PSF__` section.
psf_convolve_over_sample_size = 1

dataset = al.Imaging.from_fits(
    convolve_over_sample_size_lp=psf_convolve_over_sample_size,
    convolve_over_sample_size_pixelization=psf_convolve_over_sample_size,
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
__Extra Galaxies Noise Scaling__

Before masking, we must deal with any extra galaxies in the data: nearby galaxies (or foreground stars, or
data-reduction artefacts) whose emission is not associated with the strong lens but blends into the field. If
their light is left in the data it will contaminate the fit and bias the inferred lens model. It is too easy to
skip straight to fitting without checking for these, so we make this step an explicit part of the workflow.

To prevent extra galaxies from impacting the fit, we do not mask them entirely from the fit, which would be
analogous to making the circular mask smaller or using a more refined mask. When pixels are masked and removed
entirely from the fit, their coordinates are not used when performing ray-tracing and the light of the lens and
source galaxies in these pixels not evaluated.

Instead, the pixels are kept in the fit, but their data values are scaled to zero and their noise-map values
are increased to very large values. This means that during the fit, these pixels contribute negligibly to the
likelihood, and therefore do not impact the lens model.

This approach is used because for certain types of modeling approaches, like a pixelized source reconstruction,
masking regions of the image in a way that removes their image pixels entirely from the fit can produce
discontinuities in the pixelization. This can lead to unexpected systematics and unsatisfactory results.

The dataset includes a faint extra galaxy, and a `mask_extra_galaxies.fits` covering it is shipped with the
dataset (created by the simulator). If you are fitting your own data with an extra galaxy, you must either:

 - Create a `mask_extra_galaxies.fits` for it using the data-preparation tools (the GUI
   `autolens_workspace/*/imaging/data_preparation/gui/mask_extra_galaxies.py`, or the manual
   `autolens_workspace/*/imaging/data_preparation/examples/optional/mask_extra_galaxies.py`), then load it
   as below; or
 - Shrink the circular mask below so the extra galaxy lies outside it and is removed from the fit entirely.

After scaling, the extra galaxy's pixels have their data set to zero and noise-map increased, making their
signal-to-noise effectively zero.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,  # `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

aplt.subplot_imaging_dataset(dataset=dataset)

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
aplt.plot_array(array=dataset.data, title="Image Data With Mask Applied")

"""
The mask is also used to compute a `Grid2D`, where the (y,x) arc-second coordinates are only computed in unmasked 
pixels within the masks' circle. 

As shown in the previous overview example, this grid will be used to perform lensing calculations when fitting the
data below.
"""
aplt.plot_grid(grid=dataset.grid, title="Grid2D of Masked Dataset")

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
aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Tracer  Image")

"""
We now use a `FitImaging` object to fit this tracer to the dataset. 

The fit creates a `model_image` which we fit the data with, which includes performing the step of blurring the tracer`s 
image with the imaging dataset's PSF. We can see this by comparing the tracer`s image (which isn't PSF convolved) and 
the fit`s model image (which is).
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
aplt.fits_array(
    array=lens_subtracted_image,
    file_path=dataset_path / "lens_subtracted_data.fits",
    overwrite=True,
)

"""
Fin.
"""
