"""
Plots: Grid2DPlotter
====================

This example illustrates how to plot a `Grid2D` data structure using an `Grid2DPlotter`.

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
__Grid__

Lets create a simple uniform grid.
"""
grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1)

"""
__Figures__

We now pass the grid to a `Grid2DPlotter` and call the `figure` method.
"""
grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
We can easily ray-trace grids using a `MassProfile` and plot them with a `Grid2DPlotter`.
"""
mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.2), einstein_radius=1.0
)
deflections = mass_profile.deflections_yx_2d_from(grid=grid)

lensed_grid = grid.grid_2d_via_deflection_grid_from(deflection_grid=deflections)

grid_plotter = aplt.Grid2DPlotter(grid=lensed_grid)
grid_plotter.figure_2d()

"""
__Include__

A `Grid2D` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(origin=True)

grid_plotter = aplt.Grid2DPlotter(grid=lensed_grid, include_2d=include)
grid_plotter.figure_2d()

"""
Finish.
"""
