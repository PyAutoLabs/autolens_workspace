"""
Plots: MaskScatter
==================

This example illustrates how to customize the mask of plotted data.

Although a mask is a 2D array of values, it is actually plotted as a `Grid2D` of (y,x) coordinates corresponding to the
centre of every pixel at the edge of the mask. The mask is therefore plotted using the `GridScatter` object described
in `autolens_workspace.plot.mat_wramat_plot.wrap.base`.

The `MaskScatter` object serves the purpose is allowing us to uniquely customize the appearance of any mask on a plot.

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
We will also need the mask we will plot on the figure, which we associate with the image.
"""
mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = al.Array2D(values=data.native, mask=mask)

"""
The `Array2D` includes its mask as an internal property, meaning we can plot it via an `Include2D` object.
"""
include = aplt.Include2D(mask=True)
array_plotter = aplt.Array2DPlotter(array=masked_image_2d, include_2d=include)
array_plotter.figure_2d()

"""
The appearance of the mask is customized using a `Scatter` object.

To plot the mask this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
mask_scatter = aplt.MaskScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(mask_scatter=mask_scatter)

array_plotter = aplt.Array2DPlotter(
    array=masked_image_2d, mat_plot_2d=mat_plot, include_2d=include
)
array_plotter.figure_2d()


"""
To plot the mask manually, we can pass it into a` Visuals2D` object. 

This means we don't need to create the `masked_image` array we used above.
"""
visuals = aplt.Visuals2D(mask=mask)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
Finish.
"""
