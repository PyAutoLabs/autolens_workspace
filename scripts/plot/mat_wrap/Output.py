"""
Plots: Output
=============

This example illustrates how to output a PyAutoLens figure or subplot.

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
We can specify the output of the figure using the `Output` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.show.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.savefig.html

Below, we specify that the figure should be output as a `.png` file, with the name `example.png` in the `plot/plots` 
folder of the workspace.
"""
output = aplt.Output(
    path=path.join("notebooks", "plot", "plots"), filename="example", format="png"
)

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can specify a list of output formats, such that the figure is output to all of them.
"""
output = aplt.Output(
    path=path.join("notebooks", "plot", "plots"),
    filename="example",
    format=["png", "pdf"],
)

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
This `Output` object does not display the figure on your computer screen, bypassing this to output the `.png`. This is
the default behaviour of PyAutoLens plots, but can be manually specified using the `format-"show"`
"""
output = aplt.Output(format="show")

mat_plot = aplt.MatPlot2D(output=output)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
