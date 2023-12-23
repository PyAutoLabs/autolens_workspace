"""
Plots: Axis
===========

This example illustrates how to customize the Matplotlib figure's axis in figures and subplots.

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
We can customize the figure using the `Axis` matplotlib wrapper object which wraps the following method(s):

 plt.axis: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axis.html
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

axis = aplt.Axis(extent=[-1.0, 1.0, -1.0, 1.0])

mat_plot = aplt.MatPlot2D(axis=axis)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
