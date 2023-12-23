"""
Plots: Text
===========

This example illustrates how to customize the text that appears on a figure or subplot displayed in PyAutoLens.

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
We can customize the text using the `Text` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.text.html

We manually specify the text using the input `s` and locations `x` and `y`.
"""
text = aplt.Text(s="Example text", x=0, y=0, color="r", fontsize=20)

mat_plot = aplt.MatPlot2D(text=text)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
By passing a list of `Text` objects we can plot multiple text extracts.
"""
text_0 = aplt.Text(s="Example text 0", x=0, y=0, color="r", fontsize=20)
text_1 = aplt.Text(s="Example text 1", x=10, y=10, color="b", fontsize=20)

mat_plot = aplt.MatPlot2D(text=[text_0, text_1])

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
