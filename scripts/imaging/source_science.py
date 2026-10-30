"""
Source Science
==============

Source science focuses on studying the highly magnified properties of the background lensed source galaxy (or galaxies).

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations using Sersic parametric sources on imaging data, which
is conceptually the simplest case for source science calculations and a good introduction to the topic.

__Contents__

- **Simulated Dataset:** We load and plot the `simple__no_lens_light` example dataset, which is simulated imaging of a.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Source Values:** Source science calculations for real lenses are performed using the best-fitting model inferred.
- **Source Flux:** A key quantity for a source galaxy is its total flux, which can be used to compute magnitudes (see.
- **Source Magnification:** The overall magnification of the source is estimated as the ratio of total surface brightness in.
- **Tracer:** Lens modeling returns a `max_log_likelihood_tracer`, which is likely the object you have at hand to.
- **Magnification At A Point:** The signed magnification `mu = 1 / det(A)` at chosen (y,x) image-plane coordinates, computed via the Hessian.
- **Area Magnification:** Each lensed image's area divided by the area of the source region it comes from.
- **Parametric Source Models:** If your lens modeling uses a parametric source model (e.g.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

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

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

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

aplt.subplot_imaging_dataset(dataset=dataset)

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
)

field = al.MassField(
    redshift=0.5,
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

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

"""
By plotting the image of the tracer, we confirm it looks identical to the simulated dataset but does not have
CCD imaging features such as noise or blurring from a PSF.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

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
__Magnification At A Point__

The magnifications above are the overall magnification of the source: a ratio of fluxes integrated over the
source's full light distribution. Lens modeling codes also commonly quote the magnification at a single (y,x)
coordinate in the image-plane, for example at a lensed image, a compact clump in an arc or a transient.

This is computed from the deflection angles of the lens model via the Hessian, which gives the lensing Jacobian `A`
at each coordinate, with the magnification `mu = 1 / det(A)`.

The point magnification is signed, and its sign is the parity of the lensed image at that coordinate: positive for
images at minima (or maxima) of the arrival-time surface and negative for saddle-point images, which are
mirror-reversed relative to the source. Its absolute value is how much a point-like source at that location is
magnified. It is therefore distinct from the source magnification computed above, which averages the magnification
over the extended source and is always positive. Close to a critical curve `det(A)` goes to zero, so the point
magnification becomes very large and changes quickly with the coordinate.

Below, we compute it at three (y,x) coordinates on the lensed arcs of the simulated dataset.

For a real lens, pass the `max_log_likelihood_tracer` of your lens model and coordinates measured from your data.
For a multi-plane system (for example a cluster with sources at several redshifts), the magnification must be
computed up to the plane of the source of interest with `plane_j`; see the `Trap (b)` section of
`autolens_workspace/*/guides/advanced/multi_plane.py`.
"""
point_grid = al.Grid2DIrregular(values=[(0.3, 1.7), (-1.5, 0.2), (0.0, -1.7)])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

magnification_at_points = lens_calc.magnification_2d_via_hessian_from(grid=point_grid)

print(f"Magnification At Points (signed): {magnification_at_points}")
print(f"Magnification At Points (absolute): {abs(magnification_at_points)}")


"""
__Area Magnification__

The magnifications above weight every part of the source by its brightness. Cluster lensing studies often quote a
different, purely geometric quantity for each lensed image: its area magnification, the area of the image (for
example an arc) divided by the area of the region of the source plane it comes from.

Each small piece of an image, of area `dA`, comes from a piece of the source of area `dA / |mu|`, where `mu` is the
point magnification at that location computed above. Adding these up over the image's pixels gives the source-plane
area, and dividing the image's area by it gives the area magnification (equations 6 and 7 of the LEGGOS II paper,
arXiv:2606.20804):

    mu_area = A_image / A_source,    A_source = sum over pixels of A_pixel / |mu|

This needs only the lens model and an outline of the image, not a model of the source's light, and the source-plane
area measures the size of the part of the source that forms the image.

The calculation must be done one image at a time: a mask covering several images of the same source would count the
source once per image. `image_masks_from` splits a mask into its separate images. Images are separated by critical
curves, across which the magnification changes sign, so the mask is split into connected regions of the same sign.

Below we outline the arcs as the pixels where the lensed source is brighter than 10% of its peak, split them into
images and compute each image's area magnification and source-plane area. For a real lens you would draw this mask on
the data instead, for example with `autolens_workspace/*/imaging/data_preparation/gui/mask_arcs.py`, and use the
`max_log_likelihood_tracer` of your lens model.
"""
arc_grid = al.Grid2D.from_mask(mask=mask)

lensed_source_image = tracer.image_2d_list_from(grid=arc_grid)[-1]

arc_mask = al.Mask2D(
    mask=np.asarray(lensed_source_image.native) < 0.1 * np.max(lensed_source_image),
    pixel_scales=mask.pixel_scales,
)

image_masks = lens_calc.image_masks_from(mask=arc_mask)
area_magnifications = lens_calc.area_magnification_list_from(mask=arc_mask)

for image_mask, area_magnification in zip(image_masks, area_magnifications):
    source_plane_area = lens_calc.source_plane_area_from(mask=image_mask)

    print(
        f"Image of {np.sum(~np.asarray(image_mask))} pixels: "
        f"area magnification {area_magnification:.2f}, "
        f"source-plane area {source_plane_area:.4f} arcsec^2"
    )

"""
Every image comes from roughly the same region of the source, so their source-plane areas are similar while their
area magnifications differ. The area magnification also differs from the source magnification above, because it gives
every part of an image equal weight instead of weighting it by the source's brightness.

Two cautions. The magnification is evaluated at `sub_size` x `sub_size` points in every pixel (4 by default): increase
`sub_size` if an image lies close to a critical curve, where the magnification changes rapidly. And an image only a few
pixels wide can break into several pieces at the pixel scale, which `image_masks_from` returns as separate images:
check the masks it returns, and combine the pieces of one image before computing its area magnification.
"""

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic, Multi Gaussian Expansion), the only object
you need to perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

Alternatively, as done above, you can manually set up a tracer using the lens and source galaxies inferred
by lens modeling.

Therefore, you may now wish to go to your results, extract the `max_log_likelihood_tracer`, and use it to compute
the source flux and magnification as shown above.
"""
