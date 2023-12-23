"""
Plots: PositionsScatter
=======================

This example illustrates how to customize the positions of plotted data.

Although a positions is a 2D array of values, it is actually plotted as a `Grid2D` of (y,x) coordinates corresponding to the
centre of every pixel at the edge of the positions. The positions is therefore plotted using the `GridScatter` object described
in `autolens_workspace.plot.mat_wramat_plot.wrap.base`.

The `PositionsScatter` object serves the purpose is allowing us to uniquely customize the appearance of any positions
on a plot.

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
We will also need a positions to plot on the figure, we'll set these up as a `Grid2DIrregular` of 2D (y,x) coordinates.
"""
positions = al.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

"""
To plot the positions manually, we can pass it into a` Visuals2D` object.
"""
visuals = aplt.Visuals2D(positions=positions)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
The appearance of the positions is customized using a `Scatter` object.

To plot the positions this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
positions_scatter = aplt.PositionsScatter(marker="o", c="r", s=50)

mat_plot = aplt.MatPlot2D(positions_scatter=positions_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
Finish.
"""
