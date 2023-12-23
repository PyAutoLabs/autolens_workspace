"""
Plots: AVXLine
==============

This example illustrates how to fill between two lines on a 1D Matplotlib figure and subplots.

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
We can fill between two lines line on the y vs x plot using the `FillBetween` matplotlib wrapper object which wraps 
the following method(s):

 plt.fill_between: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill_between.html
 
The input`match_color_to_yx=True` ensures the filled area is the same as the line that is plotted, as opposed
to a different color. This ensures that shaded regions can easily be paired in color to the plot when it is 
appropriate (e.g. plotted the errors of the line).
"""
y1 = al.Array1D.no_mask(values=[0.5, 1.5, 2.5, 3.5, 4.5], pixel_scales=1.0)
y2 = al.Array1D.no_mask(values=[1.5, 2.5, 3.5, 4.5, 5.5], pixel_scales=1.0)

mat_plot = aplt.MatPlot1D(
    fill_between=aplt.FillBetween(match_color_to_yx=True, alpha=0.3)
)

visuals = aplt.Visuals1D(shaded_region=[y1, y2])

yx_plotter = aplt.YX1DPlotter(y=y, x=x, mat_plot_1d=mat_plot, visuals_1d=visuals)
yx_plotter.figure_1d()

"""
Finish.
"""
