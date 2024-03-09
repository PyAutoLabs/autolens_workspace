"""
Plots: Ticks
============

This example illustrates how to customize the Ticks of a figure or subplot displayed in PyAutoLens, by
wrapping the inputs of the Matplotlib methods `plt.tick_params`, `plt.yticks` and `plt.xticks`.

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
We can customize the ticks using the `YTicks` and `XTicks matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.yticks.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xticks.html
"""
tickparams = aplt.TickParams(
    axis="y",
    which="major",
    direction="out",
    color="b",
    labelsize=20,
    labelcolor="r",
    length=2,
    pad=5,
    width=3,
    grid_alpha=0.8,
)

yticks = aplt.YTicks(alpha=0.8, fontsize=10, rotation="vertical")
xticks = aplt.XTicks(alpha=0.5, fontsize=5, rotation="horizontal")

mat_plot = aplt.MatPlot2D(tickparams=tickparams, yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
A suffix can be added to every tick, for example the arc-seconds units can be included.

This means one does not need to include the "x (arcsec)" and "y (arcsec)" labels on the plot, saving space for
publication figures (see how to remove labels in `Labels.py`.
"""
yticks = aplt.YTicks(manual_suffix='"')
xticks = aplt.XTicks(manual_suffix='"')

mat_plot = aplt.MatPlot2D(yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()


"""
Ticks and their labels can be removed altogether by the following code:
"""
tickparams = aplt.TickParams(
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
    labelright=False,
    labeltop=False,
)

mat_plot = aplt.MatPlot2D(tickparams=tickparams, yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
