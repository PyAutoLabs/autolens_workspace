"""
Plots: PatchOverlay
===================

This example illustrates how to plot and customize patches in PyAutoLens figures and subplots.

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
To plot a patch on an image, we use the `matplotlib.patches` module. In this example, we will use
the `Ellipse` patch.
"""
from matplotlib.patches import Ellipse

patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
patch_1 = Ellipse(xy=(-2.0, -3.0), height=1.0, width=2.0, angle=1.0)

"""
We input these patches into the `Visuals2D` object, which plots it over the figure.
"""
visuals = aplt.Visuals2D(patches=[patch_0, patch_1])

"""
We now plot the image with the array overlaid.
"""
array_plotter = aplt.Array2DPlotter(array=data)  # , visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the patches using the `Patcher` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/collections_api.html
"""
patch_overlay = aplt.PatchOverlay(
    facecolor=["r", "g"], edgecolor="none", linewidth=10, offsets=3.0
)

mat_plot = aplt.MatPlot2D(patch_overlay=patch_overlay)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
Finish.
"""
