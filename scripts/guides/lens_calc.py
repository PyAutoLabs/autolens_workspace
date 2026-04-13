"""
Lens Calc
=========

This guide explains the ``LensCalc`` class, which computes a comprehensive set of lensing quantities from the
deflection angles of a mass distribution.

Given any mass model (a ``MassProfile``, ``Galaxy``, or ``Tracer``), ``LensCalc`` derives:

- **Convergence** — the projected surface mass density of the lens, normalised by the critical density.
- **Shear** — the tidal stretching and squeezing of lensed images.
- **Magnification** — how much brighter (or fainter) a lensed image appears compared to the unlensed source.
- **Critical curves** — special curves in the image plane where magnification diverges to infinity.
- **Caustics** — the source-plane counterparts of critical curves, which delimit regions of multiple imaging.
- **Einstein radius** — the characteristic angular scale of the lens, derived from the critical curves.
- **Fermat potential** — the time-delay surface, whose stationary points correspond to the observed image positions.

All of these are derived from the **deflection angles** of the lens. If you are new to gravitational lensing, this
guide walks through each quantity from first principles, with equations and code examples.

__Contents__

**Units:** In this example, all quantities use the source code's internal unit coordinates, with spatial.
**Data Structures:** Arrays inspected in this example use bespoke data structures for storing arrays, grids, vectors and.
**Grids:** To describe the deflection of light, **PyAutoLens** uses `Grid2D` data structures.
**Mass Profile and Galaxy:** We create a simple elliptical isothermal mass profile and wrap it in a Galaxy.
**Tracer:** We create a two-plane Tracer from a lens and source galaxy.
**LensCalc:** We introduce the ``LensCalc`` object and how to construct one from a Tracer.
**The Lens Equation:** The fundamental equation of gravitational lensing.
**Deflection Angles:** The deflection angles are the input to every other lensing quantity.
**Hessian:** The matrix of second derivatives of the lensing potential.
**Convergence:** The projected surface mass density normalised by the critical density.
**Shear:** The tidal distortion field that stretches lensed images.
**Magnification:** How much a lensed image is brightened (or dimmed) relative to the unlensed source.
**Critical Curves and Caustics:** Where magnification formally diverges and how this maps to the source plane.
**Einstein Radius:** The characteristic angular size of a lens.
**Fermat Potential:** The time-delay surface whose extrema locate lensed images.
**Wrap Up:** Summary and pointers to further reading.

__Units__

In this example, all quantities use the source code's internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide ``guides/units/cosmology.ipynb`` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Arrays inspected in this example use bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the ``slim`` and ``native`` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the ``slim`` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points.

These are documented fully in the ``autolens_workspace/*/guides/data_structures.ipynb`` guide.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
import autolens as al
import autoarray as aa
import autolens.plot as aplt

"""
__Grids__

To describe the deflection of light, **PyAutoLens** uses `Grid2D` data structures, which are two-dimensional
Cartesian grids of (y,x) coordinates.

Below, we make a uniform Cartesian grid in units of arcseconds. This grid will be used throughout this guide
to evaluate every lensing quantity.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

"""
__Mass Profile and Galaxy__

We create a simple elliptical isothermal mass profile (`Isothermal`). This is one of the most commonly used
mass models in strong lensing — it describes a singular isothermal ellipsoid (SIE), a good first approximation
for the mass distribution of an early-type galaxy.

We then wrap it in a `Galaxy` at redshift 0.5, which represents the foreground lens galaxy.
"""
mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    einstein_radius=1.6,
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass_profile)

"""
We also create a simple source galaxy at redshift 1.0. The source does not need a mass profile — it is the
background object whose light is being lensed.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCore(
        centre=(0.0, 0.1),
        ell_comps=(0.1, 0.0),
        intensity=0.1,
        effective_radius=0.5,
    ),
)

"""
__Tracer__

A `Tracer` combines the lens and source galaxies with a cosmological model to perform ray-tracing.

Ray-tracing means computing how light rays from the source are deflected by the lens galaxy's gravity, producing
the distorted, magnified images we observe. The `Tracer` handles all of the cosmological distance calculations
behind the scenes.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, source_galaxy],
    cosmology=al.cosmo.Planck15(),
)

"""
Here is the lensed image of the source galaxy, showing the characteristic arcs and multiple images produced by
strong gravitational lensing.
"""
image = tracer.image_2d_from(grid=grid)
aplt.plot_array(array=image, title="Lensed Image of the Source Galaxy")

"""
__LensCalc__

The `LensCalc` class is a calculator that derives all secondary lensing quantities from a deflection-angle
callable. You can construct one from a `Tracer`:
"""
lens_calc = al.LensCalc.from_tracer(tracer=tracer)

"""
You can also construct one directly from a mass profile or galaxy:

    lens_calc = al.LensCalc.from_mass_obj(mass_obj=lens_galaxy)

Both approaches give you the same interface. Using `from_tracer` is recommended when your lens system has
multiple planes or when you want to include all galaxies in the deflection calculation.

Now let's walk through each lensing quantity that `LensCalc` can compute.

__The Lens Equation__

The fundamental equation of gravitational lensing relates the **observed image position** to the **true source
position**. In the simplest case of a thin lens, this is:

    beta = theta - alpha(theta)

where:

- theta is the observed (image-plane) position of a light ray, in arcseconds.
- alpha(theta) is the deflection angle — how much the light ray is bent by the lens's gravity at position theta.
- beta is the true (source-plane) position, i.e. where the source would appear if there were no lens.

If the lens is strong enough, multiple image-plane positions theta can map to the same source position beta.
This is why we see multiple images of the same source in strong lensing systems.

Everything that `LensCalc` computes starts from the deflection angles alpha(theta).

__Deflection Angles__

The deflection angles describe how much light is bent at each point in the image plane. They are a 2D vector
field — at every (y, x) coordinate, there is a deflection in both the y and x directions.

We can compute them directly from the tracer:
"""
deflections = tracer.deflections_yx_2d_from(grid=grid)

"""
The deflection angles are a `VectorYX2D` data structure with shape [N, 2], where the first column is the
y-deflection and the second column is the x-deflection.

Let's print the deflection at the central pixel:
"""
print("Deflection at centre (y, x):", deflections[0])

"""
And plot the y-component and x-component of the deflection field:
"""
deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
aplt.plot_array(array=deflections_y, title="Deflection Angles (y-component)")

deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)
aplt.plot_array(array=deflections_x, title="Deflection Angles (x-component)")

"""
These deflection angles are the only input that `LensCalc` needs. Every other quantity — convergence, shear,
magnification, critical curves, etc. — is derived from them by taking derivatives or solving equations.

__Hessian__

The **Hessian** is the 2x2 matrix of second partial derivatives of the lensing potential psi(theta).
Equivalently, it is the matrix of first partial derivatives of the deflection angles:

    H_yy = d(alpha_y) / d(theta_y)
    H_xy = d(alpha_x) / d(theta_y)
    H_yx = d(alpha_y) / d(theta_x)
    H_xx = d(alpha_x) / d(theta_x)

Written as a matrix:

    H = | H_yy  H_xy |
        | H_yx  H_xx |

The Hessian captures how the deflection angles *change* across the image plane. This is the key to understanding
how images are distorted: convergence, shear, and magnification are all simple combinations of Hessian components.

`LensCalc` computes the Hessian by **finite differences** (nudging the grid positions slightly and measuring
how the deflections change). If JAX is available, it can alternatively use automatic differentiation for
exact derivatives.
"""
hessian_yy, hessian_xy, hessian_yx, hessian_xx = lens_calc.hessian_from(grid=grid)

print("Hessian_yy at centre:", hessian_yy[0])
print("Hessian_xx at centre:", hessian_xx[0])

"""
__Convergence__

The **convergence** (kappa) is the projected surface mass density of the lens, normalised by the **critical
surface density**. It tells you how much mass is concentrated along the line of sight at each point.

Physically:

- kappa = 1 means the projected mass density equals the critical density — this is roughly the threshold
  for strong lensing.
- kappa > 1 means the lens is "super-critical" at that point.
- kappa < 1 means the lens is "sub-critical".

The convergence is computed from the Hessian as:

    kappa = 0.5 * (H_yy + H_xx)

This is the trace of the Hessian divided by 2. It measures the isotropic part of the image distortion —
convergence magnifies images uniformly (making them bigger and brighter) without changing their shape.
"""
convergence = lens_calc.convergence_2d_via_hessian_from(grid=grid)

print("Convergence at centre:", convergence[0])

convergence_array = aa.Array2D(values=convergence, mask=grid.mask)
aplt.plot_array(array=convergence_array, title="Convergence (kappa)")

"""
The convergence computed this way (via the Hessian) is independent of any analytic formula — it works for
any mass distribution, as long as you can compute deflection angles.

__Shear__

The **shear** (gamma) describes the tidal stretching of lensed images. Unlike convergence (which magnifies
images isotropically), shear distorts images *anisotropically* — it stretches them along one axis and
compresses them along the perpendicular axis.

Shear has two components:

    gamma_1 = 0.5 * (H_xx - H_yy)
    gamma_2 = H_xy

The total shear magnitude is:

    |gamma| = sqrt(gamma_1^2 + gamma_2^2)

Physically, the shear direction tells you the orientation of the tidal stretching: images near a lens are
elongated tangentially (forming the characteristic arcs of strong lensing), while images far from the lens
experience weaker, more radially oriented distortion.
"""
shear = lens_calc.shear_yx_2d_via_hessian_from(grid=grid)

print("Shear gamma_2 at centre:", shear[0, 0])
print("Shear gamma_1 at centre:", shear[0, 1])
print("Shear magnitude at centre:", shear.magnitudes[0])

"""
__Magnification__

The **magnification** (mu) tells you how much brighter (or fainter) a lensed image appears compared to the
unlensed source. It is defined as the inverse of the determinant of the **lensing Jacobian matrix**:

    A = I - H = | 1 - H_yy   -H_xy  |
                | -H_yx    1 - H_xx  |

    mu = 1 / det(A) = 1 / [(1 - H_yy)(1 - H_xx) - H_xy * H_yx]

Equivalently, using convergence and shear:

    mu = 1 / [(1 - kappa)^2 - |gamma|^2]

Key points:

- |mu| > 1 means the image is magnified (brighter and larger than the unlensed source).
- |mu| < 1 means the image is demagnified.
- mu < 0 means the image has flipped parity (it is a mirror image of the source).
- Where det(A) = 0, the magnification diverges to infinity — these special locations are the **critical curves**.
"""
magnification = lens_calc.magnification_2d_from(grid=grid)

print("Magnification at centre:", magnification[0])

magnification_array = aa.Array2D(values=magnification, mask=grid.mask)
aplt.plot_array(array=magnification_array, title="Magnification (mu)")

"""
The magnification map shows extremely high values near the critical curves, where images are stretched into
the bright arcs that make strong lensing systems so visually striking.

__Critical Curves and Caustics__

**Critical curves** are closed curves in the image plane where the magnification formally diverges (det(A) = 0).

There are two types:

- **Tangential critical curves** — found where the tangential eigenvalue (1 - kappa - |gamma|) = 0. These
  are the ones that produce the bright, highly magnified arcs seen in strong lensing systems.

- **Radial critical curves** — found where the radial eigenvalue (1 - kappa + |gamma|) = 0. These produce
  fainter, radially oriented counter-images.

`LensCalc` finds critical curves by evaluating the eigenvalues on a fine grid and tracing the zero-contours
using a marching-squares algorithm.
"""
tangential_critical_curves = lens_calc.tangential_critical_curve_list_from(grid=grid)
radial_critical_curves = lens_calc.radial_critical_curve_list_from(grid=grid)

print("Number of tangential critical curves:", len(tangential_critical_curves))
print("Number of radial critical curves:", len(radial_critical_curves))

"""
**Caustics** are the source-plane images of the critical curves. They are found by ray-tracing each critical
curve through the lens equation (subtracting the deflection angles):

    caustic = critical_curve - alpha(critical_curve)

Caustics divide the source plane into regions with different numbers of images. A source inside the tangential
caustic produces multiple (typically 4) images, while a source outside produces fewer (typically 2).
"""
tangential_caustics = lens_calc.tangential_caustic_list_from(grid=grid)
radial_caustics = lens_calc.radial_caustic_list_from(grid=grid)

"""
Let's plot the convergence map. The critical curves trace the boundary between highly magnified and weakly
magnified regions.
"""
convergence_for_plot = tracer.convergence_2d_from(grid=grid)
aplt.plot_array(array=convergence_for_plot, title="Convergence with Critical Curves")

"""
__Einstein Radius__

The **Einstein radius** is the characteristic angular scale of a strong lens. It is defined as the radius of
the circle that encloses the same area as the tangential critical curve:

    theta_E = sqrt(A_crit / pi)

where A_crit is the area enclosed by the tangential critical curve.

This is sometimes called the "effective Einstein radius" in the literature. For a circular lens, the tangential
critical curve is a perfect circle and the Einstein radius equals its geometric radius. For an elliptical lens,
the critical curve is not circular, so the Einstein radius is an effective average.

The Einstein radius sets the scale of the lensing system — the separation between multiple images, the size
of the arcs, and the enclosed mass are all closely related to it.
"""
einstein_radius = lens_calc.einstein_radius_from(grid=grid)

print("Einstein radius:", einstein_radius, "arcsec")

"""
The angular Einstein mass (in arcseconds squared) is:

    M_E = pi * theta_E^2

To convert this to physical mass (e.g. solar masses), you need the critical surface density, which depends
on the cosmological distances to the lens and source. See the ``guides/units/cosmology.ipynb`` guide.
"""
einstein_mass = lens_calc.einstein_mass_angular_from(grid=grid)

print("Angular Einstein mass:", einstein_mass, "arcsec^2")

"""
__Fermat Potential__

The **Fermat potential** (also called the time-delay surface or arrival-time surface) is a scalar field in the
image plane that encodes the light travel time from source to observer via each image-plane position.

It is given by:

    phi(theta) = 0.5 * |theta - beta|^2 - psi(theta)

where:

- theta is the image-plane position.
- beta is the source-plane position (= theta - alpha(theta)).
- psi(theta) is the lensing potential (the scalar potential whose gradient gives the deflection angles).

The first term, 0.5 * |theta - beta|^2, is the **geometric delay** — the extra path length due to the
bending of light. The second term, psi(theta), is the **gravitational (Shapiro) delay** — the slowing of
light in the gravitational potential of the lens.

Fermat's principle tells us that observed images form at the stationary points (extrema and saddle points) of
this surface. This is a powerful result: it means the positions of lensed images are determined by the topology
of the Fermat potential.

The *differences* in the Fermat potential between image positions are proportional to the time delays between
images. Measuring these time delays (e.g. from a variable quasar) can constrain the Hubble constant.
"""
fermat_potential = lens_calc.fermat_potential_from(grid=grid)

fermat_array = aa.Array2D(values=fermat_potential, mask=grid.mask)
aplt.plot_array(array=fermat_array, title="Fermat Potential")

"""
The geometric delay term alone can also be inspected:
"""
geometric_delay = lens_calc.time_delay_geometry_term_from(grid=grid)

geometric_array = aa.Array2D(values=geometric_delay, mask=grid.mask)
aplt.plot_array(array=geometric_array, title="Geometric Time Delay Term")

"""
__Wrap Up__

This guide introduced the `LensCalc` class and the key lensing quantities it computes:

1. **Deflection angles** — the bending of light by the lens's gravity.
2. **Hessian** — the matrix of second derivatives, from which all other quantities are derived.
3. **Convergence** — the projected mass density (isotropic magnification).
4. **Shear** — the tidal distortion (anisotropic stretching).
5. **Magnification** — the brightness and size change of lensed images.
6. **Critical curves and caustics** — where magnification diverges, and the corresponding source-plane boundaries.
7. **Einstein radius** — the characteristic angular scale of the lens.
8. **Fermat potential** — the time-delay surface whose stationary points give image positions.

For further reading:

- ``guides/tracer.py`` — ray-tracing and image formation.
- ``guides/galaxies.py`` — working with individual galaxy components.
- ``guides/units/cosmology.ipynb`` — converting to physical units.
- ``guides/data_structures.py`` — the ``Array2D``, ``Grid2D`` and ``VectorYX2D`` data structures.

The `LensCalc` class is also available via `al.LensCalc` — it lives in ``PyAutoGalaxy/autogalaxy/operate/lens_calc.py``.
"""
