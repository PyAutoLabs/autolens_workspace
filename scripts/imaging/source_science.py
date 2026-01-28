"""
Source Science
==============

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations using Sersic parametric sources on imaging data, which
is conceptually the simplest case for source science calculations and a good introduction to the topic.
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
__Simulated Dataset__

We load and plot the `simple__no_lens_light` example dataset, which is simulated imaging of a strong lens
that we will use to demonstrate source science caluculations.
"""
dataset_name = "simple__no_lens_light"
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

We apply a 3.0 arcsecond circular mask and apply it to the `Imaging` object.

Source science calculations are typically performed on masked datasets to ensure only the lensed source is used
in the calculations.
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
__Source Values__

Source science calculations for real lenses are performed using the best-fitting model inferred from a dataset, 
and this example demonstrates how to use this below.

However, we for simplicity, we demonstrate these calculations using the Sersic source model used to simulate the dataset, 
which we refer to as the "true" source model. When analysing real strong lenses, a true underlying model is not known, 
but for simulated datasets it is.

This allows us to illustrate the calculations in a way that does not depend on the specific details of the data or 
on assumptions about how the lens model is inferred.

The `tracer` below corresponds to the same tracer used to simulate the `simple__no_lens_light` dataset, and therefore 
represents the true source model. We also include the 2D grid of (y,x) coordinates which simulate the dataset.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

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
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
By plotting the image of the tracer, we confirm it looks identical to the simulated dataset but does not have
CCD imaging features such as noise or blurring from a PSF.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
__Source Flux__

A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see 
`autolens_workspace/*/guides/units/flux`) example for more details on this).

The most simple way to compute the total flux of a light profile is to create a grid of (y,x) coordinates over which
we compute the image of the light profile, and then sum the image. 

The units of the light profile `intensity` are the units of the data the light profile was fitted to. In this example
we will assume everything is in electrons per second (`e- s^-1`), which is typical for Hubble Space Telescope imaging data.
"""
print(f"Source Galaxy's Intensity {source_galaxy.bulge.intensity} e- s^-1")

"""
The total flux, in units of `e- s^-1` , is computed by summing the image of the light profile over all pixels.

Note that we can use a `grid` of any shape and pixel scale here, the important thing is that it is so large
and high enough resolution that it captures all the light from the light profile.

Note that we are using the source galaxy's true light profile, which corresponds to its emission in the source-plane.
For real datasets, we have to infer this via lens modeling.
"""
grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

print(f"Total Source Flux: {total_flux} e- s^-1")

"""
Below, we will compare how this true source flux compares to the inferred source fluxes we compute using different
source modeling techniques (e.g. parametric and pixelized source models). Converting the flux to magnitudes or
other quantities used for tasks like SED fitting is described in the `autolens_workspace/*/guides/units/flux` example.

__Source Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and 
total surface brightness in the source-plane.

Note that the surface brightness is different to the total flux above, as surface brightness is flux per unit area. 
We therefore explicitly mention how area folds into the calculation below.

To ensure the magnification is stable and that we resolve all source emission in both the image-plane and source-plane 
we use a very high resolution grid, higher than we used to compute the total flux above.
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

"""
We repeat our calculation of the source's total flux in the source-plane using this higher resolution grid, note
that we do not take the area into account, the reason for this is explained below.
"""
image = source_galaxy.bulge.image_2d_from(grid=grid)

total_source_plane_flux = np.sum(image)  # in units e- s^-1 as summed over pixels

"""
We now need the total flux of the lensed source in the image-plane, that is how much flux we measure after
gravitational lensing.

To calculation this, we first ray-trace the grid above from the image-plane to the source-plane using the tracer
and then pass it to the source galaxy's light profile to compute the lensed image.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

source_plane_grid = traced_grid_list[1]

lensed_source_image = source_galaxy.bulge.image_2d_from(grid=source_plane_grid)

total_image_plane_flux = np.sum(
    lensed_source_image
)  # in units e- s^-1 as summed over pixels

"""
We now take the ratio of the total image-plane flux to source-plane flux to estimate the magnification.

Because both fluxes were computed on grids with the same total area and area per pixel, we do not need to
explicitly account for area in this calculation. This is because the area terms cancel out when taking the ratio.
Were the grid areas different, we would need to include area terms in the calculation.
"""
source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Magnification: {source_magnification}")

"""
__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to compute
source science calculations for real datasets.

The code below shows how using a tracer, composed of any combination of lens and source galaxies, we can
compute the source flux and magnification. It reproduces the calculations above.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

lensed_source_image = tracer.planes[1].image_2d_from(grid=source_plane_grid)
source_plane_image = tracer.planes[1].image_2d_from(grid=image_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)
total_source_plane_flux = np.sum(source_plane_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Total Flux via Tracer: {total_source_plane_flux} e- s^-1")
print(f"Source Magnification via Tracer: {source_magnification}")

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic, Multi Gaussian Expansion), the only object
you need to perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

Alternatively, as done above, you can manually set up a tracer using the lens and source galaxies inferred
by lens modeling.

Therefore, you may now wish to go to your results, extract the `max_log_likelihood_tracer`, and use it to compute
the source flux and magnification as shown above.
"""
