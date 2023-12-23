"""
Plots: VectorYXQuiver
========================

This example illustrates how to plot and customize vector fields in PyAutoLens figures and subplots.

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
We need a `VectorField` to plot over the image. We make a simple example of a vector field below.
"""
vectors = al.VectorYX2DIrregular(
    values=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)

"""
To plot the vector field manually, we can pass it into a` Visuals2D` object.
"""
visuals = aplt.Visuals2D(vectors=vectors)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can customize the appearance of the vectors using the `VectorYXQuiver` matplotlib wrapper object which wraps 
the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
"""
quiver = aplt.VectorYXQuiver(
    headlength=1,
    pivot="tail",
    color="w",
    linewidth=10,
    units="width",
    angles="uv",
    scale=None,
    width=0.5,
    headwidth=3,
    alpha=0.5,
)

mat_plot = aplt.MatPlot2D(vector_yx_quiver=quiver)

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=mat_plot, visuals_2d=visuals
)
array_plotter.figure_2d()

"""
Finish.
"""
