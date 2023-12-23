"""
Plots: BorderScatter
====================

This example illustrates how to customize the border of plotted data.

A border is the `Grid2D` of (y,x) coordinates at the centre of every pixel at the border of a mask. A border is defined
as a pixel that is on an exterior edge of a mask (e.g. it does not include the inner pixels of an annular mask).

The `BorderScatter` object serves the purpose is allowing us to uniquely customize the appearance of any border on
a plot.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
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
First, lets load an example Hubble Space Telescope image of a real strong lens as an `Array2D`.
"""
dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
data_path = path.join(dataset_path, "data.fits")
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
We will also need a mask whose border we will plot on the figure, which we associate with the image.
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = al.Array2D(values=data.native, mask=mask)

"""
The `Array2D` includes a its border as an internal property, meaning we can plot it via an `Include2D` object.
"""
include = aplt.Include2D(border=True)
array_plotter = aplt.Array2DPlotter(array=masked_image_2d, include_2d=include)
array_plotter.figure_2d()

"""
The appearance of the border is customized using a `BorderScatter` object.

To plot the border this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
border_scatter = aplt.BorderScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(border_scatter=border_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, include_2d=include
)
array_plotter.figure_2d()

"""
To plot the border manually, we can pass it into a` Visuals2D` object.

This means we don't need to create the `masked_image` array we used above.
"""
visuals = aplt.Visuals2D(border=mask.derive_grid.border)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
Finish.
"""
