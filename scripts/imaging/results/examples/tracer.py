"""
Results: Tracer
===============

This tutorial inspects an inferred model using the `Tracer` object inferred by the non-linear search.
This allows us to visualize and interpret its results.

This tutorial focuses on explaining how to use the inferred tracer to compute results as numpy arrays and only
briefly discusses visualization.

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
dataset_name = "lens_sersic"
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
            al.Galaxy, redshift=0.5, bulge=al.lp.Sersic, mass=al.mp.Isothermal
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=100,
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
__Log10__

The light and masss distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer,
    grid=mask.derive_grid.all_false_sub_1,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
tracer_plotter.figures_2d(image=True)

"""
__Attributes__

Printing individual attributes of the max log likelihood tracer gives us access to the inferred parameters of the
lens and source galaxies.

The tracer contains the galaxies as both a list and an instance of the model used to fit it. This means we can
access the same values in two ways, either indexing the galaxies list index or by the name used in model composition.

It can be difficult to track which galaxy is which index in the list, so it is recommended to use the model
composition to access the galaxies.
"""
print(f"Einstein Radius via list index = {tracer.galaxies[0].mass.einstein_radius}")
print(
    f"Einstein Radius via model composition = {tracer.galaxies.lens.mass.einstein_radius}"
)

"""
__Lensing Quantities__

The maximum log likelihood tracer contains a lot of information about the inferred model.

For example, by passing it a 2D grid of (y,x) coordinates we can return a numpy array containing its 2D image. This
includes the lens light and lensed source images.

Below, we use the grid of the `imaging` to computed the image on, which is the grid used to fit to the data.
"""
image = tracer.image_2d_from(grid=dataset.grid)

"""
__Data Structures Slim / Native__

The image above is returned as a 1D numpy array. 

**PyAutoLens** includes dedicated functionality for manipulating this array, for example mapping it to 2D or
performing the calculation on a high resolution sub-grid which is then binned up. 

This uses the data structure API, which is described in the `results/examples/data_structures.py` example. This 
tutorial will avoid using this API, but if you need to manipulate results in more detail you should check it out.
"""
print(image.slim)

"""
__Grid Choices__

We can input a different grid, which is not masked, to evaluate the image everywhere of interest. We can also change
the grid's resolution from that used in the model-fit.

The examples uses a grid with `shape_native=(3,3)`. This is much lower resolution than one would typically use to 
perform ray tracing, but is chosen here so that the `print()` statements display in a concise and readable format.
"""
grid = al.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.1)

image = tracer.image_2d_from(grid=grid)

print(image.slim)

"""
__Sub Gridding__

A grid can also have a sub-grid, defined via its `sub_size`, which defines how each pixel on the 2D grid is split 
into sub-pixels of size (`sub_size` x `sub_size`). 

The calculation below shows how to use a sub-grid and bin it up, full details of the API for this calculation
are given in the `results/examples/data_structure.py` example.
"""
grid_sub = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1, sub_size=2)

image = tracer.image_2d_from(grid=grid_sub)

print(image.binned)

"""
__Positions Grid__

We may want the image at specific (y,x) coordinates.

We can use an irregular 2D (y,x) grid of coordinates for this. The grid below evaluates the image at:

- y = 1.0, x = 1.0.
- y = 1.0, x = 2.0.
- y = 2.0, x = 2.0.
"""
grid_irregular = al.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

image = tracer.image_2d_from(grid=grid_irregular)

print(image)

"""
__Scalar Lensing Quantities__

The tracer has many scalar lensing quantities, which are all returned using an `Array2D` and therefore use the same 
interface as images, described above.

For example, we can compute the `Tracer`'s convergence using all of the grids above.
"""
convergence_2d = tracer.convergence_2d_from(grid=dataset.grid)
print(convergence_2d)

convergence_2d = tracer.convergence_2d_from(grid=grid_sub)
print(convergence_2d)

convergence_2d = tracer.convergence_2d_from(grid=grid_irregular)
print(convergence_2d)

"""
This is the convergence of every galaxy in the tracer summed together. It may not be appropriate if your lens model 
performs multi-plane ray-tracing (e.g. there are more than 2 redshifts containing galaxies). Later results tutorials
provide tools that are more appropriate for multi-plane tracers.

There are other scalar quantities accessible via the tracer (those not familiar with strong lensing mathematical 
formalism may not recognise what these quantities are -- don't worry about it for now!):
"""
potential_2d = tracer.potential_2d_from(grid=dataset.grid)

tangential_eigen_value = tracer.tangential_eigen_value_from(grid=dataset.grid)
radial_eigen_value = tracer.radial_eigen_value_from(grid=dataset.grid)


"""
A 2D magnification map is available, which using only the ray-tracing and therefore mass model quantities how much
light rays are focus at a given point in the image-plane.

If you are studying a strongly lensed source galaxy and want to know how much the galaxy itself is magnified, the
magnification below is not of too much use too you. In the result tutorial `galaxies.py` we explain how the 
magnification of the source can be quantified.
"""
magnification_2d = tracer.magnification_2d_from(grid=dataset.grid)

"""
__Vector Quantities__

Many lensing quantities are vectors. That is, they are (y,x) coordinates that have 2 values representing their
magnitudes in both the y and x directions.

These quantities also have a dedicated data structure which is described fully in 
the `results/examples/data_structure.py` example.

The most obvious of these is the deflection angles, which are used throughout lens modeling to ray-trace grids
from the image-plane to the source-plane via a lens galaxy mass model.

To indicate that a quantities is a vector, **PyAutoLens** uses the label `_yx`
"""
deflections_yx_2d = tracer.deflections_yx_2d_from(grid=dataset.grid)

"""
For vector quantities the has shape `2`, corresponding to the y and x vectors respectively.
"""
print(deflections_yx_2d[0, :])

"""
The `VectorYX2D` object has a built in method to return the magnitude of each vector, which is a scalar quantity
and therefore returned using a 1D Numpy array.
"""
deflection_magnitudes_2d = deflections_yx_2d.magnitudes
print(deflection_magnitudes_2d)

"""
__Other Vector Lensing Quantities__

The tracer has other vector lensing quantities, which use the same interface described above.
"""
shear_yx_2d = tracer.shear_yx_2d_via_hessian_from(grid=dataset.grid)

"""
__Other Quantities__

Many more quantities are shown below.

A full description of each can be found in the docstring of the source code of each function:

   https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/operate/deflections.py
"""
tangential_critical_curve = tracer.tangential_critical_curve_list_from(
    grid=dataset.grid
)

radial_critical_curve = tracer.radial_critical_curve_list_from(grid=dataset.grid)

tangential_caustic = tracer.tangential_caustic_list_from(grid=dataset.grid)

radial_caustic = tracer.radial_caustic_list_from(grid=dataset.grid)

### You should be able to comment this out and it work fine ###

# area_within_tangential_critical_curve = (
#     tracer.tangential_critical_curve_area_list_from(grid=dataset.grid)
# )
#
# einstein_radius = tracer.einstein_radius_from(grid=dataset.grid)
#
# einstein_mass_angular = tracer.einstein_mass_angular_from(grid=dataset.grid)

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the tracer changes for models with similar log likelihoods. Below, we create and plot
the tracer of the 100th last accepted model by Nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

tracer = al.Tracer(galaxies=instance.galaxies)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=mask.derive_grid.all_false_sub_1
)
tracer_plotter.subplot_tracer()

"""
__Wrap Up__

This tutorial explained how to compute the results of an inferred model from a tracer. 

It omitted a number of tasks we may want to do, for example:

 - We only created an image-plane image of the lens and lensed source. How do I view the source galaxy in the source 
   plane?

 - We could only compute the image, convergence, potential and other properties of the entire `Tracer` object. What 
   if I want these quantities for specific galaxies in the tracer?

 - How do I estimate errors on these quantities?
 
 - How do I properly account for multi-plane ray-tracing effects?
 
The example `results/examples/galaxies.py` illustrates how to perform these more detailed calculations.
"""
