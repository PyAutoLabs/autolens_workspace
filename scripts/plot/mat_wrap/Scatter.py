"""
Plots: Scatter
==============

This example illustrates how to plot and customize (y,x) grids of coordinates in PyAutoLens figures and subplots.

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
The appearance of a (y,x) `Grid2D` of coordinates is customized using `Scatter` objects. To illustrate this, we will 
customize the appearance of the (y,x) origin on a figure using an `OriginScatter` object.

To plot a (y,x) grids of coordinates (like an origin) these objects wrap the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
origin_scatter = aplt.OriginScatter(marker="o", s=50)

mat_plot = aplt.MatPlot2D(origin_scatter=origin_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, include_2d=aplt.Include2D(origin=True), mat_plot_2d=mat_plot
)
array_plotter.figure_2d()

"""
There are numerous (y,x) grids of coordinates that PyAutoLens plots. For example, in addition to the origin,
there are grids like the multiple images of a strong lens, a source-plane grid of traced coordinates, etc.

All of these grids are plotted using a `Scatter` object and they are described in more detail in the 
`plot/include_2d` example scripts. 
"""
