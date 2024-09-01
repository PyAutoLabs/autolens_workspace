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

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoLens** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/guides/data_structures.ipynb` guide.

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
__Grids__

To describe the luminous emission of galaxies, **PyAutoGalaxy** uses `Grid2D` data structures, which are 
two-dimensional Cartesian grids of (y,x) coordinates. 

Below, we make and plot a uniform Cartesian grid:
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
)

grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

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

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.subplot_tracer()

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
The `LightProfilePlotter` makes it straight forward to extract and plot an individual light profile component.
"""
bulge_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.planes[0][0].bulge, grid=grid
)
bulge_plotter.figures_2d(image=True)

"""
__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
bulge_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.planes[0][0].bulge,
    grid=grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
bulge_plotter.figures_2d(image=True)

"""
__Galaxies__

Above, we extract the `bulge` and `disk` light profiles. 

We can just as easily extract each `Galaxy` and use it to perform the calculations above. Note that because the 
lens galaxy contains both the `bulge` and `disk`, the `image` we create below contains both components (and is therefore
the same as `tracer.image_2d_from(grid=grid)`:
"""
lens = tracer.planes[0][0]

lens_image_2d = lens.image_2d_from(grid=grid)
lens_convergence_2d = lens.convergence_2d_from(grid=grid)

"""
We can also use the `GalaxyPlotter` to plot the lens galaxy, for example a subplot of each individual light profile 
image and mass profile convergence.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens, grid=grid)
galaxy_plotter.subplot_of_light_profiles(image=True)
galaxy_plotter.subplot_of_mass_profiles(convergence=True)

"""
__Source Plane Images__

We can also extract the source-plane galaxies to plot images of them.

We create a specific uniform grid to plot these images. Because this grid is an image-plane grid, the images of the
source are their unlensed source-plane images (we show have to create their lensed images below). 
"""
grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.05)

source_0 = tracer.planes[1][0]
source_1 = tracer.planes[1][1]

# source_0 = tracer.galaxies.source_0
# source_1 = tracer.galaxies.source_1

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_0, grid=grid)
galaxy_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_1, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
__Tracer Composition__

Lets quickly summarize what we've learnt by printing every object in the tracer:
"""
print(tracer)
print(tracer.planes[0])  # image plane
print(tracer.planes[1])  # source plane
print(tracer.planes[0][0])  # lens galaxy in image plane
print(tracer.planes[1][0])  # source galaxy 0 in source plane
print(tracer.planes[1][1])  # source galaxy 1 in source plane
print(tracer.planes[0][0].mass)  # lens galaxy mass profile
print(tracer.planes[1][0].bulge)  # source galaxy 0 bulge
print(tracer.planes[1][1].bulge)  # source galaxy 1 bulge
print()

"""
__Lensed Grids and Images__

In order to plot source-plane images that are lensed we can compute traced grids from the tracer.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

"""
The first grid in the list is the image-plane grid (and is identical to `grid`) whereas the second grid has
had its coordinates deflected via the tracer's lens galaxy mass profiles.
"""
image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

"""
We can use the `source_plane_grid` to created an image of both lensed source galaxies.
"""
source_0 = tracer.planes[1][0]
source_0_image_2d = source_0.image_2d_from(grid=source_plane_grid)

source_0 = tracer.planes[1][1]
source_1_image_2d = source_1.image_2d_from(grid=source_plane_grid)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_0, grid=source_plane_grid)
galaxy_plotter.figures_2d(image=True)

"""
___Source Magnification__

The overall magnification of the source is estimated as the ratio of total surface brightness in the image-plane and 
total surface brightness in the source-plane.

To ensure the magnification is stable and that we resolve all source emission in both the image-plane and source-plane 
we use a very high resolution grid (in contrast to calculations above which used the lower resolution masked imaging 
grids).

(If an inversion is used to model the source a slightly different calculation is performed which is discussed in
result tutorial 6.)
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

"""
We now compute the image of each plane using the above two grids, where the ray-traced `source_plane_grid`
creates the image of the lensed source and `image_plane_grid` creates the source-plane image of the source.

(By using `tracer.planes[1].image_2d_from`, as opposed to `tracer.image_2d_from`, we ensure that only source-plane
emission is included and that lens light emission is not).
"""
lensed_source_image_2d = tracer.planes[1].image_2d_from(grid=source_plane_grid)
source_plane_image_2d = tracer.planes[1].image_2d_from(grid=image_plane_grid)

"""
The `source_plane_grid` and `image_plane_grid` grids below were created above by ray-tracing the
first one to create the other. 

They therefore evaluate the lensed source and source-plane emission on grids with the same total area.

When computing magnifications, care must always be taken to ensure the areas in the image-plane and source-plane
are properly accounted for.
"""
print(lensed_source_image_2d.total_area)
print(source_plane_image_2d.total_area)

"""
Because their areas are the same, we can estimate the magnification by simply taking the ratio of total flux.
"""
source_magnification_2d = np.sum(lensed_source_image_2d) / np.sum(source_plane_image_2d)

"""
__One Dimensional Quantities__

We have made two dimensional plots of galaxy images, grids and convergences.

We can also compute all these quantities in 1D, for inspection and visualization.

For example, from a light profile or galaxy we can compute its `image_1d`, which provides us with its image values
(e.g. luminosity) as a function of radius.
"""
lens = tracer.planes[0][0]
image_1d = lens.image_1d_from(grid=grid)
print(image_1d)

source_bulge = tracer.planes[1][0].bulge
image_1d = source_bulge.image_1d_from(grid=grid)
print(image_1d)

"""
How are these 1D quantities from an input 2D grid? 

From the 2D grid a 1D grid is compute where:

 - The 1D grid of (x,) coordinates are centred on the galaxy or light profile and aligned with the major-axis. 
 - The 1D grid extends from this centre to the edge of the 2D grid.
 - The pixel-scale of the 2D grid defines the radial steps between each coordinate.

If we input a larger 2D grid, with a smaller pixel scale, the 1D plot adjusts accordingly.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.04)
image_1d = lens.image_1d_from(grid=grid)
print(image_1d.shape)
print(image_1d)

grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02)
image_1d = lens.image_1d_from(grid=grid)
print(image_1d.shape)
print(image_1d)

"""
We can alternatively input a `Grid1D` where we define the (x,) coordinates we wish to evaluate the function on.
"""
grid_1d = al.Grid1D.uniform_from_zero(shape_native=(10000,), pixel_scales=0.01)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens, grid=grid)

galaxy_plotter.figures_1d(image=True, convergence=True)

"""
Fin.
"""
