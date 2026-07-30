"""
Plots: Objects
==============

This example illustrates how to plot each key PyAutoLens object — light profiles, mass profiles,
galaxies and tracers — figure by figure.

Every object follows the same pattern: a quantity is computed via the object's method
(e.g. `image_2d_from()`, `convergence_2d_from()`, `deflections_yx_2d_from()`) and the resulting
array or grid is passed to `aplt.plot_array()` or `aplt.plot_grid()`.

For an introduction to the plotting API itself (customization, output, config defaults, overlays,
subplots) refer to `guides/plot/start_here.py`. For plotting datasets and fits (e.g. `FitImaging`),
refer to the `plot.py` example of each dataset package (e.g. `scripts/imaging/plot.py`).

__Contents__

- **Setup:** Set up the galaxies and tracer used throughout this example.
- **Light Profile:** A light profile image is computed via `image_2d_from()` and plotted with `aplt.plot_array()`.
- **Mass Profile:** Mass profile quantities (convergence, potential, deflection angles) are computed and plotted individually.
- **Galaxy:** A galaxy's image and mass quantities are computed and plotted with `aplt.plot_array()`.
- **Tracer:** Tracer quantities unique to ray-tracing (deflections, magnification, source-plane image) are computed and plotted.
- **Lensed Grids:** A grid ray-traced via the tracer's deflection angles is plotted with `aplt.plot_grid()`.
- **1D Profiles:** 1D radial profiles are computed on projected or radial grids and plotted with matplotlib.

__Setup__

Set up standard objects used throughout this example.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
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
__Light Profile__

A light profile image is computed via `image_2d_from()` and plotted with `aplt.plot_array()`.
"""
bulge = tracer.galaxies[0].bulge
aplt.plot_array(array=bulge.image_2d_from(grid=grid), title="Bulge Image")

"""
__Mass Profile__

Mass profile quantities are computed and plotted individually.
"""
mass = tracer.galaxies[0].mass
aplt.plot_array(array=mass.convergence_2d_from(grid=grid), title="Mass Convergence")
aplt.plot_array(array=mass.potential_2d_from(grid=grid), title="Mass Potential")

"""
Deflection angles are returned as a grid of (y,x) values. To plot the y or x component as a
2D array, wrap the corresponding column in an `Array2D` using the grid's mask.
"""
import autoarray as aa

mass_deflections = mass.deflections_yx_2d_from(grid=grid)
aplt.plot_array(
    array=aa.Array2D(values=mass_deflections.slim[:, 0], mask=grid.mask),
    title="Mass Deflections Y",
)
aplt.plot_array(
    array=aa.Array2D(values=mass_deflections.slim[:, 1], mask=grid.mask),
    title="Mass Deflections X",
)

"""
__Galaxy__

A galaxy's image and mass quantities are computed and plotted with `aplt.plot_array()`.

These sum the quantities of all light and mass profiles the galaxy contains.
"""
galaxy = tracer.galaxies[0]
aplt.plot_array(array=galaxy.image_2d_from(grid=grid), title="Galaxy Image")
aplt.plot_array(array=galaxy.convergence_2d_from(grid=grid), title="Galaxy Convergence")

"""
__Tracer__

The tracer's image, convergence and potential are plotted following the same pattern as the
examples above (see `guides/plot/start_here.py`). Quantities unique to ray-tracing are shown
below.

The tracer's deflection angles are plotted component by component, like a mass profile's.
"""
deflections_yx = tracer.deflections_yx_2d_from(grid=grid)

aplt.plot_array(
    array=aa.Array2D(values=deflections_yx.slim[:, 0], mask=grid.mask),
    title="Deflections Y",
)
aplt.plot_array(
    array=aa.Array2D(values=deflections_yx.slim[:, 1], mask=grid.mask),
    title="Deflections X",
)

"""
The magnification map is computed via a `LensCalc` object.
"""
lens_calc = al.LensCalc.from_tracer(tracer=tracer)
aplt.plot_array(array=lens_calc.magnification_2d_from(grid=grid), title="Magnification")

"""
The source-plane image (plane index 1) is accessed via the image list.
"""
aplt.plot_array(
    array=tracer.image_2d_list_from(grid=grid)[1],
    title="Source Plane Image",
)

"""
__Lensed Grids__

A grid ray-traced via the tracer's deflection angles is plotted with `aplt.plot_grid()`,
showing where the image-plane coordinates land in the source plane.
"""
deflections = tracer.deflections_yx_2d_from(grid=grid)
lensed_grid = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)
aplt.plot_grid(grid=lensed_grid, title="Lensed Grid")

"""
__1D Profiles__

1D radial profiles are computed using a projected 2D grid and plotted with matplotlib directly.

There is no dedicated 1D plotting function — matplotlib is used for full control over 1D figures.
"""
grid_2d_projected = grid.grid_2d_radial_projected_from(
    centre=galaxy.bulge.centre, angle=bulge.angle()
)

image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
Using a radial grid of (y,x) coordinates along the x-axis plots the 1D radial profile.
"""
radii = np.arange(10000) * 0.01
grid_radial = al.Grid2DIrregular(values=[(0.0, r) for r in radii])
image_1d = bulge.image_2d_from(grid=grid_radial)

plt.plot(radii, image_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
