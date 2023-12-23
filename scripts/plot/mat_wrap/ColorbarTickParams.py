"""
Plots: ColorbarTickParams
=========================

This example illustrates how to customize the ticks on a Colorbar in PyAutoLens figures and subplots.

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
We can customize the colorbar ticks using the `ColorbarTickParams` matplotlib wrapper object which wraps the 
following method of the matplotlib colorbar:

 https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
"""
colorbar_tickparams = aplt.ColorbarTickParams(
    axis="both",
    reset=False,
    which="major",
    direction="in",
    length=2,
    width=2,
    color="r",
    pad=0.1,
    labelsize=10,
    labelcolor="r",
)

mat_plot = aplt.MatPlot2D(colorbar_tickparams=colorbar_tickparams)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The colorbar tick parameters can be removed altogether with the following:
"""
colorbar_tickparams = aplt.ColorbarTickParams(
    bottom=False, top=False, left=False, right=False
)

mat_plot = aplt.MatPlot2D(colorbar_tickparams=colorbar_tickparams)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
