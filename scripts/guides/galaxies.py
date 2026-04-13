"""
Galaxies
========

In the guide `tracer.py`, we inspected the results of a `Tracer` and computed the overall properties of the
lens model's image, convergence and other quantities.

However, we did not compute the individual properties of each galaxy. For example, we did not compute an image of the
source galaxy on the source plane or compute individual quantities for each mass profile.

This tutorial illustrates how to compute these more complicated results. We therefore fit a slightly more complicated
lens model, where the lens galaxy's light is composed of two components (a bulge and disk) and the source-plane
comprises two galaxies.

__Contents__

**Units:** In this example, all quantities use the source code's internal unit coordinates, with spatial.
**Data Structures:** Arrays inspected in this example use bespoke data structures for storing arrays, grids, vectors and.
**Grids:** To describe the luminous emission of galaxies, **PyAutoGalaxy** uses `Grid2D` data structures.
**Tracer:** We first set up a tracer with a lens galaxy and two source galaxies, which we will use to.
**Individual Lens Galaxy Components:** We are able to create an image of the lens galaxy as follows, which includes the emission of both.
**Log10:** The light distributions of galaxies are closer to a log10 distribution than a linear one.

__Units__

In this example, all quantities use the source code's internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Arrays inspected in this example use bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/guides/data_structures.ipynb` guide.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Grids__

To describe the luminous emission of galaxies, **PyAutoGalaxy** uses `Grid2D` data structures, which are 
two-dimensional Cartesian grids of (y,x) coordinates. 

Below, we make and plot a uniform Cartesian grid:
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

aplt.plot_grid(grid=grid, title="")

"""
__Tracer__

We first set up a tracer with a lens galaxy and two source galaxies, which we will use to illustrate how to extract
individual galaxy images.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=2.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=0.1,
        effective_radius=1.6,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.25, 0.15),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=0.7,
        effective_radius=0.7,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.7, -0.5),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=0.2,
        effective_radius=1.6,
        sersic_index=3.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

aplt.subplot_tracer(tracer=tracer, grid=grid)

"""
__Individual Lens Galaxy Components__

We are able to create an image of the lens galaxy as follows, which includes the emission of both the lens galaxy's
`bulge` and `disk` components.
"""
image = tracer.image_2d_from(grid=grid)

"""
In order to create images of the `bulge` and `disk` separately, we need to extract each individual component from the 
tracer. 

To do this, we first use the tracer's `planes` attribute, which is a list of all `Planes` objects in the tracer. 

This list is in ascending order of plane redshift, such that `planes[0]` is the image-plane and `planes[1]` is the 
source-plane. Had we modeled a multi-plane lens system there would be additional planes at each individual redshift 
(the redshifts of the galaxies in the model determine at what redshifts planes are created).
"""
image_plane = tracer.planes[0]
source_plane = tracer.planes[1]

"""
Each plane contains a list of galaxies, which are in order of how we specify them in the `collection` above.

In order to extract the `bulge` and `disk` we therefore need the lens galaxy, which we can extract from 
the `image_plane` and print to make sure it contains the correct light profiles.
"""
print(image_plane)

lens_galaxy = image_plane[0]

print(lens_galaxy)

"""
Finally, we can use the `lens_galaxy` to extract the `bulge` and `disk` and make the image of each.
"""
bulge = lens_galaxy.bulge
disk = lens_galaxy.disk

bulge_image_2d = bulge.image_2d_from(grid=grid)
disk_image_2d = disk.image_2d_from(grid=grid)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(bulge_image_2d.slim[0])
print(disk_image_2d.slim[0])

"""
It is more concise to extract these quantities in one line of Python.

The way to think about index accessing of `planes`, as shown below is as follows:

- The first index, `planes[0]` accesses the first plane (the image-plane).
- The second index, `planes[0][0]` accesses the first galaxy in the first plane (the lens galaxy).
"""
bulge_image_2d = tracer.planes[0][0].bulge.image_2d_from(grid=grid)

"""
The `aplt.plot_array` makes it straight forward to extract and plot an individual light profile component.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `plot_array`/`subplot_\*` object has an input `use_log10`, which will do this automatically when we call the `plot_array` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
