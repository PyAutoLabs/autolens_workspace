"""
Plots: Array2DPlotter
=====================

This example illustrates how to plot an `Array2D` data structure using an `Array2DPlotter`.

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
__Dataset__

First, lets load an example image of of a strong lens as an `Array2D`.
"""
dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
data_path = path.join(dataset_path, "data.fits")
data = al.Array2D.from_fits(file_path=data_path, hdu=0, pixel_scales=0.03)

"""
__Figures__

We now pass the array to an `Array2DPlotter` and call the `figure` method.
"""
array_plotter = aplt.Array2DPlotter(array=data)
array_plotter.figure_2d()

"""
__Include__

An `Array2D` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to illustrate
plotting its mask and border below).
"""
include = aplt.Include2D(origin=True, mask=True, border=True)

mask = al.Mask2D.circular_annular(
    shape_native=data.shape_native,
    pixel_scales=data.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image_2d = al.Array2D(values=data.native, mask=mask)

array_plotter = aplt.Array2DPlotter(array=masked_image_2d, include_2d=include)
array_plotter.figure_2d()

"""
Finish.
"""
