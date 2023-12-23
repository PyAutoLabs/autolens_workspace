"""
Plots: Cmap
===========

This example illustrates how to customize the Colormap in PyAutoLens figures and subplots.

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
We can customize the colormap using the `Cmap` matplotlib wrapper object which wraps the following method(s):

 colors.Linear: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
 colors.LogNorm: https://matplotlib.org/3.3.2/tutorials/colors/colormapnorms.html
 colors.SymLogNorm: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.colors.SymLogNorm.html
        
The colormap is used in various functions that plot images with a `cmap`, most notably `plt.imshow`:

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
 
First, lets plot the image using a linear colormap which uses a `colors.Normalize` object.
"""
cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=0.0, vmax=1.0)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can instead use logarithmic colormap (this wraps the `colors.LogNorm` matplotlib object).
"""
cmap = aplt.Cmap(cmap="hot", norm="log", vmin=0.0, vmax=2.0)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
We can use a symmetric log norm (this wraps the `colors.SymLogNorm` matplotlib object).
"""
cmap = aplt.Cmap(
    cmap="twilight",
    norm="symmetric_log",
    vmin=0.0,
    vmax=1.0,
    linthresh=0.05,
    linscale=0.1,
)

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The diverge normalization ensures the colorbar is centred around 0.0, irrespective of whether vmin and vmax are input.
"""
cmap = aplt.Cmap(cmap="twilight", norm="diverge")

mat_plot = aplt.MatPlot2D(cmap=cmap)

array_plotter = aplt.Array2DPlotter(array=data, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
Finish.
"""
