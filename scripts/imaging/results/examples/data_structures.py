"""
Results: Data Structures
========================

This tutorial illustrates the data structure objects which many results quantities are stored using, which are
extensions of NumPy arrays.

These data structures are used because for different lensing calculations it is convenient to store the data in
different formats. For example, when ray-tracing a uniform grid of image-plane (y,x) coordinates, to an irregular
grid of source-plane (y,x) coordinates, the image-plane coordinates can be stored in 2D whereas the source-plane
coordinates must be stored in 1D.

These data structures use the `slim` and `native` data representations API to make it simple to map quantities from
1D dimensions to their native dimensions (e.g. a 2D grid).

It also includes functionality necessary for performing calculations on a sub-grid, and binning this grid up to
perform more accurate calculations.

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
    )
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
__Data Structures Slim / Native__

Objects like the `Tracer` allow us to produce lens modeling quantities.

For example, by passing it a 2D grid of (y,x) coordinates we can return a numpy array containing its 2D image. 
This includes the lens light and lensed source images.

Below, we use the grid of the `imaging` to computed the image on, which is the grid used to fit to the data.
"""
image = tracer.image_2d_from(grid=dataset.grid)

"""
If we print the type of the `image` we note that it is an `Array2D`, which is a data structure that inherits 
from a numpy array but is extended to include specific functionality discussed below.
"""
print(type(image))

"""
Because the image is a numpy array, we can print its shape and see that it is 1D.
"""
print(image.shape)

"""
__Data Structure__

Why is the image stored as a 1D NumPy array? Is the image not a 2D quantity?

Every array object returned is accessible via two attributes, `native` and `slim`:

 - `slim`: an ndarray of shape [total_unmasked_pixels] which is a slimmed-down representation of the data in 1D that 
    contains only the unmasked data points (where this mask is the one used by the model-fit above).

 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels], which is the native shape of the 
    masked 2D grid used to fit the lens model. All masked pixels are assigned a value 0.0 in the `native` array.
"""
print(image.native.shape)
print(image.slim.shape)

"""
By default, all arrays in **PyAutoLens** are stored as their `slim` 1D numpy array.

We can easily access them in their native format.
"""
print(image[0:2])
print(image.slim[0:2])
print(image.native[10:12, 10:12])

"""
__Grid Choices__

We can input a different grid, which is not masked, to evaluate the image anywhere of interest. We can also change
the grid's resolution from that used in the model-fit.

The examples uses a grid with `shape_native=(3,3)`. This is much lower resolution than one would typically use to 
perform ray tracing, but is chosen here so that the `print()` statements display in a concise and readable format.
"""
grid = al.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.1)

image = tracer.image_2d_from(grid=grid)

print(image.slim)
print(image.native)

"""
__Sub Gridding__

A grid can also have a sub-grid, defined via its `sub_size`, which defines how each pixel on the 2D grid is split 
into sub-pixels of size (`sub_size` x `sub_size`). 

These additional sub-pixels are used to perform calculations more accurately. For example, for the 2D image the
values can be computed at every sub-pixel coordinate and binned-up, as opposed to computing the image only at the
centre of each image pixel. 

This approximates more closely how light is observed on a telescope, where it is the full surface brightness 
distribution of the source over the pixel that is observed.

The `sub_shape_native` and `sub_shape_slim` properties of the grid show that it has many additional coordinates
corresponding to the 4x4 grid of sub-pixels in each image pixel.
"""
grid_sub = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1, sub_size=2)

print(grid_sub.sub_shape_native)
print(grid_sub.sub_shape_slim)

"""
The image computed using this grid does not have a `native` shape (5,5), but instead shape (20, 20). This is because 
each image pixel has been split into a 4x4 sub pixel (e.g. 4 * 5 = 20):
"""
image = tracer.image_2d_from(grid=grid_sub)

print(image.native.shape)
print(image.shape)

"""
To estimate the image on our original 5x5 grid, we can use the `binned` property which bins up every 4x4 grid
of sub-pixels in each image pixel.
"""
print(image.binned.slim)
print(image.binned.native)

"""
__Slim and Native Grids__

Now we are familiar with `slim` and `native` datasets, it is worth noting that `Grid`'s also use this structure.

They can be thought of as behaving analogously to vectors, albeit grids do not contains (y,x) vectors on a (y,x)
grids of coordinates, but are simply the (y,x) grid of coordinates by itself.
"""
grid_sub = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1, sub_size=2)

print(grid_sub.slim)
print(grid_sub.native)
print(grid_sub.binned.slim)
print(grid_sub.binned.native)

"""
A more detailed description of sub-gridding is provided in the optional **HowToLens** tutorial
`autolens_workspace/*/howtolens/chapter_optional/tutorial_sub_grids.ipynb`.

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
__Vector Quantities__

Many lensing quantities are vectors. That is, they are (y,x) coordinates that have 2 values representing their
magnitudes in both the y and x directions.

The most obvious of these is the deflection angles, which are used throughout lens modeling to ray-trace grids
from the image-plane to the source-plane via a lens galaxy mass model.

To indicate that a quantities is a vector, **PyAutoLens** uses the label `_yx`
"""
deflections_yx_2d = tracer.deflections_yx_2d_from(grid=dataset.grid)

"""
If we print the type of the `deflections_yx` we note that it is a `VectorYX2D`.
"""
print(type(deflections_yx_2d))

"""
Unlike the scalar quantities above, which were a 1D numpy array in the `slim` representation and a 2D numpy array in 
the `native` representation, vectors are 2D in `slim` and 3D in `native`.
"""
print(deflections_yx_2d.slim.shape)
print(deflections_yx_2d.native.shape)

"""
For vector quantities the has shape `2`, corresponding to the y and x vectors respectively.
"""
print(deflections_yx_2d.slim[0, :])

"""
The role of the terms `slim` and `native` can be thought of in the same way as for scalar quantities. 

For a scalar, the `slim` property gives every scalar value as a 1D ndarray for every unmasked pixel. For a vector we 
still get an ndarray of every unmasked pixel, however each entry now contains two entries: the vector of (y,x) values. 

For a `native` property these vectors are shown on an image-plane 2D grid where again each pixel
contains a (y,x) vector.

Like we did for the convergence, we can use whatever grid we want to compute a vector and use sub-gridding to estimate
values more precisely:
"""
grid = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1)

deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid)

print(deflections_yx_2d.slim)
print(deflections_yx_2d.native)

grid_sub = al.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1, sub_size=2)

deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid_sub)

print(deflections_yx_2d.binned.slim)
print(deflections_yx_2d.binned.native)

grid_irregular = al.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid_irregular)

print(deflections_yx_2d)

"""
Finish.
"""
