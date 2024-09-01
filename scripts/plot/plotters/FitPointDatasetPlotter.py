"""
Plots: PointDatasetPlotter
==========================

This example illustrates how to plot a `PointDataset` dataset using an `PointDatasetPlotter`.

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

First, lets load an example strong lens `PointDataset` object.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "point_source", dataset_name)

dataset = al.from_json(
    file_path=path.join(dataset_path, "point_dataset.json"),
)

"""
__Fit__

We now fit it with a `Tracer` to create a `FitPointDataset` object.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=0.8)
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

fit = al.FitPointDataset(dataset=dataset, tracer=tracer, solver=solver)

"""
__Figures__

We now pass the FitPointDataset to a `FitPointDatasetPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
fit_plotter = aplt.FitPointDatasetPlotter(fit=fit)
fit_plotter.figures_2d(positions=True, fluxes=True)

"""
Finish.
"""
