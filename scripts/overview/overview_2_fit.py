"""
Overview: Fit
-------------

**PyAutoLens** uses `Tracer` objects to represent a strong lensing system. Now, we`re going use these objects to
fit `Imaging` data of a strong lens.

"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Loading Data__

We we begin by loading the strong lens dataset `simple__no_lens_light` from .fits files, which is the dataset 
we will use to demonstrate fitting.

The `autolens_workspace` comes distributed with simulated images of strong lenses (an example of how these simulations
are made can be found in the `simulate.py` example, with all simulator scripts located in `autolens_workspac/simulators`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We can use the `ImagingPlotter` to plot the image, noise-map and psf of the dataset.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True, noise_map=True, psf=True)

"""
The `ImagingPlotter` also contains a subplot which plots all these properties simultaneously.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We now mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.

We use a ``Mask2D`` object, which for this example is a 3.0" circular mask.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
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

Following the previous overview example, we can make a tracer from a collection of `LightProfile`, `MassProfile` 
and `Galaxy` objects.

The combination of `LightProfile`'s and `MassProfile`'s below is the same as those used to generate the simulated 
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

This subplot is probably the most important visualization output by **PyAutoLens**, and is something you should
anticipate seeing a lot of!
"""
fit_plotter.subplot_fit()

"""
Most importantly, the ``FitImaging`` object also provides us with a ``log_likelihood``, a single value quantifying 
how good the tracer fitted the dataset.

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
__Wrap Up__

If you are unfamiliar with data and model fitting, and unsure what terms like 'residuals', 'chi-squared' or 'likelihood' 
mean, we'll explain all in chapter 1 of the **HowToLens** lecture series. Checkout the
`tutorials <https://pyautolens.readthedocs.io/en/latest/tutorials/howtolens.html>`_ section of the readthedocs!
"""
