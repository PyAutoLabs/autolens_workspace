"""
Plots: AVXLine
==============

This example illustrates how to plot vertical lines on a 1D Matplotlib figure and subplots.

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
First, lets create 1D data for the plot.
"""
y = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)
x = al.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0, 5.0], pixel_scales=1.0)

"""
We now pass this y and x data to a `YX1DPlotter` and call the `figure` method.
"""
yx_plotter = aplt.YX1DPlotter(y=y, x=x)
yx_plotter.figure_1d()

"""
We can plot a vertical line on the y vs x plot using the `AXVLine` matplotlib wrapper object which wraps the 
following method(s):

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
# mat_plot = aplt.MatPlot1D(axv)

yx_plotter = aplt.YX1DPlotter(y=y, x=x)
yx_plotter.figure_1d()

"""
Finish.
"""
