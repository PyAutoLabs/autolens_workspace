"""
Plots: GridScatter
==================

This example illustrates how to plot a 2D `Grid2D` of (y,x) coordinates over PyAutoLens figures and subplots.

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
We next need the 2D `Grid2D` we overlay. We'll create a uniform grid at a coarser resolution than our dataset.
"""
grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)

"""
We input this `Grid2D` into the `Visuals2D` object, which plots it over the figure.
"""
visuals = aplt.Visuals2D(grid=grid)

"""
We now plot the image with the grid overlaid.
"""
array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We customize the grid's appearance using the `GridScatter` `matplotlib wrapper object which wraps the following method(s): 

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
grid_scatter = aplt.GridScatter(c="r", marker=".", s=1)

mat_plot = aplt.MatPlot2D(grid_scatter=grid_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
Finish.
"""
