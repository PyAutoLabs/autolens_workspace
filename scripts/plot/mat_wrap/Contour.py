"""
Plots: Contours
===============

This example illustrates how to customize the contours in PyAutoLens figures and subplots.

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
__Light__

Create a light profile which we will use to plot contours over a image map.
"""
light = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=0.1,
    effective_radius=1.0,
    sersic_index=4.0,
)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

image = light.image_2d_from(grid=grid)

"""
We can customize the contour using the `Contour` matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.contour.html
"""
contour = aplt.Contour(colors="k")

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Levels__

By default, the contour levels are computed automatically from the minimum and maximum values of the array. 

They are then plotted in 10 intervals spaced evenly in log10 values between these limits.

The number of contour levels and use of linear spacing can be manually input.
"""
contour = aplt.Contour(colors="k", total_contours=5, use_log10=False)

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
The levels can also be manually specified using the `manual_levels` input.
"""
contour = aplt.Contour(manual_levels=[0.1, 0.5, 10.0])

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()

"""
__Values__

By default, the value of each contour is shown on the figure.

This can be disabled using the `include_values` input.
"""
contour = aplt.Contour(include_values=False)

mat_plot = aplt.MatPlot2D(contour=contour)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot)
array_plotter.figure_2d()


"""
Finish.
"""
