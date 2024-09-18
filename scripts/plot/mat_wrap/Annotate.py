"""
Plots: Annotate
===============

This example illustrates how to customize annotations that appears on a figure or subplot displayed in PyAutoLens.
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
We can customize the annotations using the `Annotate` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.annotate.html

We manually specify the annotation using the input `s` and locations `x` and `y`, which means a solid white line
wil. appear.
"""
annotate = aplt.Annotate(
    text="",
    xy=(2.0, 2.0),
    xytext=(3.0, 2.0),
    arrowprops=dict(arrowstyle="-", color="w"),
)

mat_plot = aplt.MatPlot2D(annotate=annotate)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By passing a list of `Annotate` objects we can plot multiple text extracts.
"""
annotate_0 = aplt.Annotate(
    text="",
    xy=(2.0, 2.0),
    xytext=(3.0, 2.0),
    arrowprops=dict(arrowstyle="-", color="w"),
)
annotate_1 = aplt.Annotate(
    text="",
    xy=(1.0, 1.0),
    xytext=(2.0, 1.0),
    arrowprops=dict(arrowstyle="-", color="r"),
)

mat_plot = aplt.MatPlot2D(annotate=[annotate_0, annotate_1])

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
