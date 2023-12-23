"""
Plots: HalfLightRadiusAXVLine
=============================

This example illustrates how to plot the half-light radius of a `LightProfile` on 1D figures of its properties.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
First, lets create a simple `LightProfile` which we'll plot.
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=1.0,
    effective_radius=0.8,
    sersic_index=4.0,
)

"""
We also need the 2D grid the `LightProfile` is evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
We now pass the light profile and grid to a `LightProfilePlotter` and call the `figures_1d` methods to plot its image
as a function of radius.

The `LightProfile` includes the half-light radius as an internal property, meaning we can plot it via an `Include1D` 
object.
"""
include = aplt.Include1D(half_light_radius=True)
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=bulge, grid=grid, include_1d=include
)
light_profile_plotter.figures_1d(image=True)

"""
The appearance of the half-light radius is customized using a `HalfLightRadiusAXVLine` object.

To plot the half-light radius as a vertical line this wraps the following matplotlib method:

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
half_light_radius_axvline = aplt.HalfLightRadiusAXVLine(
    linestyle="-.", c="r", linewidth=20
)

mat_plot = aplt.MatPlot1D(half_light_radius_axvline=half_light_radius_axvline)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=bulge, grid=grid, mat_plot_1d=mat_plot, include_1d=include
)
light_profile_plotter.figures_1d(image=True)

"""
To plot the half-light radius manually, we can pass it into a` Visuals1D` object.
"""
visuals = aplt.Visuals1D(half_light_radius=bulge.half_light_radius)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=bulge, grid=grid, visuals_1d=visuals
)
light_profile_plotter.figures_1d(image=True)

"""
Finish.
"""
