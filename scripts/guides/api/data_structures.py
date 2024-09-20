"""
Data Structures
===============

This tutorial illustrates the data structure objects which data and results quantities are stored using, which are
extensions of NumPy arrays.

These data structures are used because for different lensing calculations it is convenient to store the data in
different formats. For example, when ray-tracing a uniform grid of image-plane (y,x) coordinates, to an irregular
grid of source-plane (y,x) coordinates, the image-plane coordinates can be stored in 2D (because the grid is uniform)
whereas the source-plane coordinates must be stored in 1D (because after lensing it is irregular).

These data structures use the `slim` and `native` data representations API to make it simple to map quantities from
1D dimensions to their native dimensions.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__API__

We discuss in detail why these data structures and illustrate their functionality below.

However, we first create the three data structures we'll use in this example, to set expectations for what they do.

We create three data structures:

 - `Array2D`: A 2D array of data, which is used for storing an image, a noise-map, etc. 

 - `Grid2D`: A 2D array of (y,x) coordinates, which is used for ray-tracing.

 -`VectorYX2D`: A 2D array of vector values, which is used for deflection angles, shear and other vector fields.

All data structures are defined according to a uniform grid of coordinates and therefore they have a `pixel_scales`
input defining the pixel-to-arcssecond conversion factor of its grid. 

For example, for an image stored as an `Array2D`, it has a grid where each coordinate is the centre of each image pixel
and the pixel-scale is therefore the resolution of the image.

We first create each data structure without a mask using the `no_mask` method:
"""
arr = al.Array2D.no_mask(
    values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
)

print(arr)

grid = al.Grid2D.no_mask(
    values=[
        [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
        [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    ],
    pixel_scales=1.0,
)

print(grid)

vector_yx = al.VectorYX2D.no_mask(
    values=[
        [[5.0, -5.0], [5.0, 0.0], [5.0, 5.0]],
        [[0.0, -5.0], [0.0, 0.0], [0.0, 5.0]],
        [
            [-5.0, -5.0],
            [-5.0, 0.0],
            [-5.0, 5.0],
        ],
    ],
    pixel_scales=1.0,
)

print(vector_yx)

"""
__Grids__

We now illustrate data structures using a `Grid2D` object, which is a set of two-dimensional $(y,x)$ coordinates
(in arc-seconds) that are deflected and traced by a strong lensing system.

These are fundamental to all lensing calculations and drive why data structures are used in **PyAutoLens**.

First, lets make a uniform 100 x 100 grid of (y,x) coordinates and plot it.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Uniform 100 x 100 Grid2D"))

grid_plotter = aplt.Grid2DPlotter(grid=grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Native__

This plot shows the grid in its `native` format, that is in 2D dimensions where the y and x coordinates are plotted
where we expect them to be on the grid.

We can print values from the grid's `native` property to confirm this:
"""
print("(y,x) pixel 0:")
print(grid.native[0, 0])
print("(y,x) pixel 1:")
print(grid.native[0, 1])
print("(y,x) pixel 2:")
print(grid.native[0, 2])
print("(y,x) pixel 100:")
print(grid.native[1, 0])
print("etc.")

"""
__Slim__

Every `Grid2D` object is accessible via two attributes, `native` and `slim`, which store the grid as NumPy ndarrays 
of two different shapes:
 
 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels, 2] which is the native shape of the 
 2D grid and corresponds to the resolution of the image datasets we pair with a grid.
 
 - `slim`: an ndarray of shape [total_y_image_pixels*total_x_image_pixels, 2] which is a slimmed-down representation 
 the grid which collapses the inner two dimensions of the native ndarray to a single dimension.
"""
print("(y,x) pixel 0 (accessed via native):")
print(grid.native[0, 0])
print("(y,x) pixel 0 (accessed via slim 1D):")
print(grid.slim[0])

"""
As discussed above, the reason we need the slim representation is because when we ray-trace a grid of (y,x) coordinates
from the image-plane to the source-plane, the source-plane grid will be irregular.

The shapes of the `Grid2D` in its `native` and `slim` formats are also available, confirming that this grid has a 
`native` resolution of (100 x 100) and a `slim` resolution of 10000 coordinates.
"""
print(grid.shape_native)
print(grid.shape_slim)

"""
Neither shape above include the third index of the `Grid` which has dimensions 2 (corresponding to the y and x 
coordinates). 

This is accessible by using the standard numpy `shape` method on each grid.
"""
print(grid.native.shape)
print(grid.slim.shape)

"""
We can print the entire `Grid2D` in its `slim` or `native` form. 
"""
print(grid.native)
print(grid.slim)

"""
__Masked Data Structures__

When a mask is applied to a grid or other data structure, this changes the `slim` and `native` representations as 
follows:

 - `slim`: only contains image-pixels that are not masked, removing all masked pixels from the 1D array.
 
 - `native`: retains the dimensions [total_y_image_pixels, total_x_image_pixels], but the masked pixels have values
    of 0.0 or (0.0, 0.0).

This can be seen by computing a grid via a mask and comparing the its`shape_slim` attribute to the `pixels_in_mask` of 
the mask.
"""
mask = al.Mask2D.circular(shape_native=(100, 100), pixel_scales=0.05, radius=3.0)

grid = al.Grid2D.from_mask(mask=mask)

print("The shape_slim and number of unmasked pixels")
print(grid.shape_slim)
print(mask.pixels_in_mask)

"""
We can use the `slim` attribute to print unmasked values of the grid:
"""
print("First unmasked image value:")
print(grid.slim[0])

"""
The `native` representation of the `Grid2D` retains the dimensions [total_y_image_pixels, total_x_image_pixels], 
however the exterior pixels have values of 0.0 indicating that they have been masked.
"""
print("Example masked pixels in the grid native representation:")
print(grid.shape_native)
print(grid.native[0, 0])
print(grid.native[2, 2])

"""
__Data__

Two dimensional arrays of data are stored using the `Array2D` object, which has `slim` and `native` representations
analogous to the `Grid2D` object and described as follows:

 - `slim`: an ndarray of shape [total_unmasked_pixels] which is a slimmed-down representation of the data in 1D that 
    contains only the unmasked data points (where this mask is the one used by the model-fit above).

 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels], which is the native shape of the 
    masked 2D grid used to fit the lens model. All masked pixels are assigned a value 0.0 in the `native` array.

For example, the `data` and `noise_map` in an `Imaging` object are stored as `Array2D` objects.

We load an imaging dataset and illustrate its data structures below.   
"""
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

data = dataset.data

"""
Here is what `slim` and `native` representations of the data's first pixel look like for the `data` before masking:
"""
print("First unmasked data value:")
print(data.slim[0])
print(data.native[0, 0])

"""
By default, all arrays in **PyAutoLens** are stored as their `slim` 1D numpy array, meaning we don't need to use the
`slim` attribute to access the data.
"""
print(data[0])

"""
By applying a mask the first value in `slim` changes and the native value becomes 0.0:
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

data = dataset.data

print("First unmasked data value:")
print(data.slim[0])
print(data.native[0, 0])

"""
__Tracer__

The `Tracer` produces many lensing quantities all of which use the `slim` and `native` data structures.

For example, by passing it a 2D grid of (y,x) coordinates we can return a numpy array containing its 2D image. 
This includes the lens light and lensed source images.

Below, we use the grid that is aligned with the imaging data (e.g. where each grid coordinate is at the centre of each
image pixel) to compute the galaxy image and show its data structure.
"""
lens = al.Galaxy(
    redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)
)

source = al.Galaxy(
    redshift=1.0,
    light=al.lp.SersicCoreSph(
        centre=(0.0, 0.0),
        intensity=0.2,
        effective_radius=0.2,
        sersic_index=1.0,
        radius_break=0.025,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

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
__Irregular Structures__

We may want to perform calculations at specific (y,x) coordinates which are not tied to a uniform grid.

We can use an irregular 2D (y,x) grid of coordinates for this. The grid below evaluates the image at:

- y = 1.0, x = 1.0.
- y = 1.0, x = 2.0.
- y = 2.0, x = 2.0.
"""
grid_irregular = al.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

image = tracer.image_2d_from(grid=grid_irregular)

print(image)

"""
The result is stored using an `ArrayIrregular` object, which is a data structure that handles irregular arrays.
"""
print(type(image))

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

grid_irregular = al.Grid2DIrregular(values=[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]])

deflections_yx_2d = tracer.deflections_yx_2d_from(grid=grid_irregular)

print(deflections_yx_2d)

"""
Finish.
"""
