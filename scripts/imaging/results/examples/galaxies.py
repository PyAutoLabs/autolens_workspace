"""
Results: Galaxies
=================

In results tutorial `tracer.py`, we inspected the results of a `Tracer` and computed the overall properties of the
lens model's image, convergence and other quantities.

However, we did not compute the individual properties of each galaxy. For example, we did not compute an image of the
source galaxy on the source plane or compute individual quantities for each mass profile.

This tutorial illustrates how to compute these more complicated results. We therefore fit a slightly more complicated
lens model, where the lens galaxy's light is composed of two components (a bulge and disk) and the source-plane
comprises two galaxies.

__Plot Module__

This example uses the **PyAutoLens** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoLens** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/guides/data_structure.ipynb` guide.

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
__Model Fit__

The code below performs a model-fit using Nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple__source_x2"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=al.lp.Sersic,
            disk=al.lp.Exponential,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source_0=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
        source_1=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]_mass[sie]_source_x2[bulge]",
    unique_tag=dataset_name,
    n_live=150,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Tracer__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_tracer` which we can visualize.
"""
tracer = result.max_log_likelihood_tracer

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=mask.derive_grid.all_false_sub_1
)
tracer_plotter.subplot_tracer()

"""
__Individual Lens Galaxy Components__

We are able to create an image of the lens galaxy as follows, which includes the emission of both the lens galaxy's
`bulge` and `disk` components.
"""
image = tracer.image_2d_from(grid=dataset.grid)

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
lens_galaxy = image_plane.galaxies[0]

print(lens_galaxy)

"""
Finally, we can use the `lens_galaxy` to extract the `bulge` and `disk` and make the image of each.
"""
bulge = lens_galaxy.bulge
disk = lens_galaxy.disk

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
disk_image_2d = disk.image_2d_from(grid=dataset.grid)

"""
If you are unclear on what `slim` means, refer to the section `Data Structure` at the top of this example.
"""
print(bulge_image_2d.slim[0])
print(disk_image_2d.slim[0])

"""
It is more concise to extract these quantities in one line of Python:
"""
bulge_image_2d = tracer.planes[0].galaxies[0].bulge.image_2d_from(grid=dataset.grid)

"""
The `LightProfilePlotter` makes it straight forward to extract and plot an individual light profile component.
"""
bulge_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.planes[0].galaxies[0].bulge, grid=dataset.grid
)
bulge_plotter.figures_2d(image=True)

"""
__Alternative API__

In the first results tutorial, we used `Samples` objects to inspect the results of a model.

We saw how these samples created instances, which include a `galaxies` property that mains the API of the `Model`
creates above (e.g. `galaxies.lens.bulge`). 

We can also use this instance to extract individual components of the model.
"""
samples = result.samples

ml_instance = samples.max_log_likelihood()

bulge = ml_instance.galaxies.lens.bulge

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
print(bulge_image_2d.slim[0])

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=dataset.grid)
bulge_plotter.figures_2d(image=True)

"""
In fact, if we create a `Tracer` from an instance (which is how `result.max_log_likelihood_tracer` is created) we
can choose whether to access its attributes using each API: 
"""
tracer = result.max_log_likelihood_tracer
print(tracer.galaxies.lens.bulge)

"""
We'll use the former API from here on. 

Whilst its a bit less clear and concise, it is more representative of the internal **PyAutoLens** source code and
therefore gives a clearer sense of how the internals work.

__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
bulge_plotter = aplt.LightProfilePlotter(
    light_profile=tracer.planes[0].galaxies[0].bulge,
    grid=dataset.grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
bulge_plotter.figures_2d(image=True)

"""
__Galaxies__

Above, we extract the `bulge` and `disk` light profiles. 

We can just as easily extract each `Galaxy` and use it to perform the calculations above. Note that because the 
lens galaxy contains both the `bulge` and `disk`, the `image` we create below contains both components (and is therefore
the same as `tracer.image_2d_from(grid=dataset.grid)`:
"""
lens = tracer.planes[0].galaxies[0]

lens_image_2d = lens.image_2d_from(grid=dataset.grid)
lens_convergence_2d = lens.convergence_2d_from(grid=dataset.grid)

"""
We can also use the `GalaxyPlotter` to plot the lens galaxy, for example a subplot of each individual light profile 
image and mass profile convergence.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens, grid=dataset.grid)
galaxy_plotter.subplot_of_light_profiles(image=True)
galaxy_plotter.subplot_of_mass_profiles(convergence=True)

"""
__Source Plane Images__

We can also extract the source-plane galaxies to plot images of them.

We create a specific uniform grid to plot these images. Because this grid is an image-plane grid, the images of the
source are their unlensed source-plane images (we show have to create their lensed images below). 
"""
grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.05)

source_0 = tracer.planes[1].galaxies[0]
source_1 = tracer.planes[1].galaxies[1]

# source_0 = tracer.galaxies.source_0
# source_1 = tracer.galaxies.source_1

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_0, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_1, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Tracer Composition__

Lets quickly summarize what we've learnt by printing every object in the tracer:
"""
print(tracer)
print(tracer.planes[0])
print(tracer.planes[1])
print(tracer.planes[0].galaxies[0])
print(tracer.planes[1].galaxies[0])
print(tracer.planes[0].galaxies[0].mass)
print(tracer.planes[1].galaxies[0].bulge)
print(tracer.planes[1].galaxies[1].bulge)
print()

"""
__Lensed Grids and Images__

In order to plot source-plane images that are lensed we can compute traced grids from the tracer.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

"""
The first grid in the list is the image-plane grid (and is identical to `dataset.grid`) whereas the second grid has
had its coordinates deflected via the tracer's lens galaxy mass profiles.
"""
image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[1]

"""
We can use the `source_plane_grid` to created an image of both lensed source galaxies.
"""
source_0 = tracer.planes[1].galaxies[0]
source_0_image_2d = source_0.image_2d_from(grid=source_plane_grid)

source_0 = tracer.planes[1].galaxies[1]
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
lens = tracer.planes[0].galaxies[0]
image_1d = lens.image_1d_from(grid=grid)
print(image_1d)

source_bulge = tracer.planes[1].galaxies[0].bulge
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
__Decomposed 1D Plot__

We can make a plot containing every individual light and mass profile of a galaxy in 1D, for example showing a 
decomposition of its `bulge` and `disk`.

Every profile on a decomposed plot is computed using a radial grid centred on its profile centre and aligned with
its major-axis. Therefore 2D offsets between the components are not portray in such a figure.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens, grid=grid)
galaxy_plotter.figures_1d_decomposed(image=True)

"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light and mass models estimated via a 
model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of the model-fit. 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `einstein_radius` with errors, which are plotted as vertical lines.

Below, we manually input one hundred realisations of the lens galaxy with light and mass profiles that clearly show 
these errors on the figure.
"""
galaxy_pdf_list = [samples.draw_randomly_via_pdf().galaxies.lens for i in range(10)]

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=galaxy_pdf_list, grid=grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(
    #    image=True,
    #   convergence=True,
    #   potential=True
)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(
    # image=True,
    #  convergence=True,
    #  potential=True
)

"""
__Wrap Up__

We have learnt how to extract individual planes, galaxies, light and mass profiles from the tracer that results from
a model-fit and use these objects to compute specific quantities of each component.
"""
