"""
Units and Cosmology
===================

This tutorial illustrates how to perform unit conversions from **PyAutoLens**'s internal units (e.g. arc-seconds,
electrons per second, dimensionless mass units) to physical units (e.g. kiloparsecs, magnitudes, solar masses).

This is used on a variety of important lens model cosmological quantities for example the lens's Einstein radius and
Mass or the effective radii of the galaxies in the lens model.

__Contents__

**Errors:** To produce errors on unit converted quantities, you`ll may need to perform marginalization over.
**Tracer:** We set up a simple strong lens tracer and grid which will illustrate the unit conversion.
**Arcsec to Kiloparsec:** The majority of distance quantities in **PyAutoLens** are in arcseconds, because this means that.
**Einstein Radius:** Given a tracer, galaxy or mass profile we can compute its Einstein Radius, which is defined as the.
**Einstein Mass:** The Einstein mass can also be computed from a tracer, galaxy or mass profile.
**Convergence:** The `colorbar_convert_factor` and `colorbar_label` inputs above can also be used to convert the.

__Errors__

To produce errors on unit converted quantities, you`ll may need to perform marginalization over samples of these
converted quantities (see `results/aggregator/samples.ipynb`).
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Tracer__

We set up a simple strong lens tracer and grid which will illustrate the unit conversion functionality. 
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        einstein_radius=1.6,
    ),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Arcsec to Kiloparsec__

The majority of distance quantities in **PyAutoLens** are in arcseconds, because this means that known redshifts are
not required in order to compose the lens model.

By assuming redshifts for the lens and source galaxies we can convert their quantities from arcseconds to kiloparsecs.

Below, we compute the effective radii of the source in kiloparsecs. To do this, we assume a cosmology which 
allows us to compute the conversion factor `kpc_per_arcsec`.
"""
cosmology = al.cosmo.Planck15()

source = tracer.planes[1][0]
source_plane_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=source.redshift)
source_effective_radius_kpc = (
    source.bulge.effective_radius * source_plane_kpc_per_arcsec
)

"""
This `kpc_per_arcsec` can be used as a conversion factor between arcseconds and kiloparsecs when plotting images of
galaxies.

Below, we compute this value in both the image-plane and source-plane, and plot the images in both planes in their
respectively converted units of kilo-parsec.

This passes the plotting modules `Units` object a `ticks_convert_factor` and manually specified the new units of the
plot ticks.
"""
lens = tracer.planes[0][0]
image_plane_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=lens.redshift)


aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")


aplt.plot_grid(
    grid=tracer.traced_grid_2d_list_from(grid=grid)[-1], title="Source Plane"
)

"""
__Einstein Radius__

Given a tracer, galaxy or mass profile we can compute its Einstein Radius, which is defined as the area within the 
tangential critical curve. 

These are calculated from the functions: 

 - `einstein_radius_from`. 
 - `einstein_mass_via_tangential_critical_curve`.

Although these quantities should not depend on the grid we input, they are calculated using the input grid. Thus,
we must specify a grid which matches the scale of the lens model, which would typically be the grid of image-pixels
that we use to model our data.

Lets print the Einstein Radius, which is returned in the default internal **PyAutoLens** units of arc-seconds.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
einstein_radius = lens_calc.einstein_radius_from(grid=grid)

print("Einstein Radius (arcsec) = ", einstein_radius)

"""
If we know the redshift of the lens galaxy and assume an cosmology we can convert this to kilo-parsecs.
"""
cosmology = al.cosmo.Planck15()

kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=tracer.planes[0].redshift)
einstein_radius_kpc = einstein_radius * kpc_per_arcsec
print("Einstein Radius (kpc) = ", einstein_radius_kpc)

"""
We can also compute the Einstein radius of individual planes, galaxies and mass profiles.
"""
print(
    al.LensCalc.from_mass_obj(mass_obj=tracer.planes[0][0]).einstein_radius_from(
        grid=grid
    )
)
print(
    al.LensCalc.from_mass_obj(mass_obj=tracer.planes[0][0].mass).einstein_radius_from(
        grid=grid
    )
)

"""
__Einstein Mass__

The Einstein mass can also be computed from a tracer, galaxy or mass profile.

The default units of an Einstein mass are angular units; this is because to convert it to physical units (e.g. solar
masses) one must assume redsfhits for the lens and source galaxies.

The mass in angular units is given by: `pi * einstein_radius (arcsec) ** 2.0`
"""
einstein_mass = lens_calc.einstein_mass_angular_from(grid=grid)
print("Einstein Mass (angular) = ", einstein_mass)

"""
To convert this mass to solar masses, we need the critical surface mass density of the strong lens, which relies on 
it being a strong lens with not only a lens redshift (e.g. the redshift of the profile) but also a source redshift.

If we use the `tracer`'s galaxies for the redshifts, where the lens is at redshift 0.5 and it is lensing a source at 
redshift 1.0, we can compute its mass in solar masses.
"""
cosmology = al.cosmo.Planck15()

critical_surface_density = cosmology.critical_surface_density_between_redshifts_from(
    redshift_0=tracer.planes[0].redshift, redshift_1=tracer.planes[1].redshift
)
einstein_mass_solar_mass = einstein_mass * critical_surface_density
print("Einstein Mass (solMass) = ", einstein_mass_solar_mass)
print("Einstein Mass (solMass) = ", "{:.4e}".format(einstein_mass_solar_mass))

"""
We can compute Einstein masses of individual components:
"""
print(
    al.LensCalc.from_mass_obj(mass_obj=tracer.planes[0][0]).einstein_mass_angular_from(
        grid=grid
    )
)
print(
    al.LensCalc.from_mass_obj(
        mass_obj=tracer.planes[0][0].mass
    ).einstein_mass_angular_from(grid=grid)
)

"""
In principle, the Einstein Mass of a `Tracer` should be readily accessible in a `Tracer` object, given this contains
all of the galaxies in a strong lens system (and thus has their redshifts) as well as an input Cosmology.

However, we do not provide methods with this quantity and require that you, the user, compute the Einstein mass 
(in angular or solar masses) using examples above. This is because for systems with multiple galaxies or planes, the 
definition of an Einstein Radius / Mass become less clear. 

We feel it is better that a user explicitly computes these quantities from a `Tracer` so if it has multiple galaxies 
or planes you are aware of this.

__Brightness Units / Luminosity__

When plotting the image of a galaxy, each pixel value is also plotted in electrons / second, which is the unit values
displayed in the colorbar. 

A conversion factor between electrons per second and another unit can be input when plotting images of galaxies.

Below, we pass the exposure time of the image, which converts the units of the image from `electrons / second` to
electrons. 

Note that this input `ticks_convert_factor_values` is the same input parameter used above to convert mass plots like the 
convergence to physical units.
"""
exposure_time_seconds = 2000.0


"""
The luminosity of a galaxy is the total amount of light it emits, which is computed by integrating the light profile.
This integral is performed over the entire light profile, or within a specified radius.

Lets compute the luminosity of the source galaxy in the default internal **PyAutoLens** units of `electrons / second`.
Below, we compute the luminosity to infinite radius, which is the total luminosity of the galaxy, but one could
easily compute the luminosity within a specified radius instead.
"""
source = tracer.planes[1][0]

luminosity = source.luminosity_within_circle_from(radius=np.inf)
print("Luminosity (electrons / second) = ", luminosity)

"""
From a luminosity in `electrons / second`, we can convert it to other units, such as `Jansky` or `erg / second`. 
This can also be used to compute the magnitude of the galaxy, which is the apparent brightness of the galaxy in a
given bandpass.

This functionality is not currently implemented in **PyAutoLens**, but would be fairly simple for you to do
yourself (e.g. using the `astropy` package). If you want to contribute to **PyAutoLens**, this would be a great
first issue to tackle, so please get in touch on SLACK!

__Convergence__

The `colorbar_convert_factor` and `colorbar_label` inputs above can also be used to convert the units of mass
profiles images. 

For example, we can convert the convergence from its dimensionless lensing units to a physical surface density
in units of solar masses per kpc^2.
"""
critical_surface_density = cosmology.critical_surface_density_between_redshifts_from(
    redshift_0=tracer.planes[0].redshift, redshift_1=tracer.planes[1].redshift
)

convergence = tracer.convergence_2d_from(grid=grid)

"""
With the convergence in units of MSun / kpc^2, we can easily compute the total mass associated with it in a specifc
area.

For example, in a single pixel of convergence in these units, we can compute the mass by simply multiplying it by the
area of the pixel in kpc^2.
"""
pixel_area_kpc = (
    grid.pixel_scales[0] * grid.pixel_scales[1] * image_plane_kpc_per_arcsec**2
)

print(
    f"Total mass in central pixel: {convergence.native[convergence.shape_native[0] // 2, convergence.shape_native[1] // 2] * critical_surface_density * pixel_area_kpc} MSun"
)

"""
The total mass of the convergence map is the sum of all these masses.
"""
print(
    f"Total mass in convergence map: {np.sum(convergence * critical_surface_density * pixel_area_kpc)} MSun"
)
