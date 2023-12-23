"""
Plots: OriginScatter
====================

This example illustrates how to customize the (y,x) origin of plotted data.

The origin is plotted using the `GridScatter` object described in `autolens_workspace.plot.mat_wramat_plot.wrap.base`.
The `OriginScatter` object serves the purpose is allowing us to unique customize the appearance of an origin on a plot.

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
The `Array2D` includes its mask as an internal property, meaning we can plot it via an `Include2D` object.

As can be seen below, the origin of the data is (0.0, 0.0), which is where the black cross marking the origin
appears.
"""
include = aplt.Include2D(origin=True)

array_plotter = aplt.Array2DPlotter(array=data, include_2d=include)
array_plotter.figure_2d()

"""
The appearance of the (y,x) origin coordinates is customized using a `Scatter` object.

To plot these (y,x) grids of coordinates these objects wrap the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
 
The example script `plot/mat_wrap/Scatter.py` gives a more detailed description on how to customize its appearance.
"""
origin_scatter = aplt.OriginScatter(marker="o", s=50)

mat_plot = aplt.MatPlot2D(origin_scatter=origin_scatter)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, include_2d=include
)
array_plotter.figure_2d()

"""
To plot the origin manually, we can pass it into a` Visuals2D` object.
"""
visuals = aplt.Visuals2D(origin=al.Grid2DIrregular(values=[(1.0, 1.0)]))

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
Finish.
"""
