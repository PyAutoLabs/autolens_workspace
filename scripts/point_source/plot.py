"""
Plots: Point Source
===================

This example shows how to plot a `PointDataset` and a `FitPointDataset` fit.

Point-source data differs from imaging and interferometer data: it consists of the (y,x) positions
of a point source's multiple images (and optionally their fluxes and time delays), not a 2D image.
Plotting therefore centres on the multiple-image positions and the fit's ability to trace them back
to a common source.

For an introduction to the plotting API refer to `guides/plot/start_here.py`. The final section
documents the `Visualizer`, which outputs point-source fit figures automatically during a model-fit.

__Contents__

- **Dataset:** Load the point-source dataset of multiple-image positions.
- **Dataset Figures:** Inspect the dataset's info and plot its multiple-image positions.
- **Fit:** Set up a tracer and point solver and fit the dataset with a `FitPointDataset`.
- **Fit Figures:** Inspect the fit's residuals and plot the fit subplot.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the point-source dataset `simple`, which contains the positions of a lensed quasar's multiple
images.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not (dataset_path / "point_dataset_positions_only.json").exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulator.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=Path(dataset_path, "point_dataset_positions_only.json"),
)

"""
__Dataset Figures__

The dataset's `info` summarizes its name, positions and noise-map values. The name (`point_0`) is
what pairs the dataset to a point source in a lens model (see `scripts/point_source/fit.py`).
"""
print(dataset.info)

"""
The multiple-image positions are a `Grid2DIrregular`, plotted with `aplt.plot_grid()`.
"""
aplt.plot_grid(grid=dataset.positions, title="Multiple Image Positions")

"""
__Fit__

To fit the positions we compose a tracer whose mass model and point source match the true simulated
values, and a `PointSolver` which solves the lens equation to find where the point source's multiple
images appear.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.8,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0, point_0=al.ps.PointFlux(centre=(0.0, 0.0), flux=0.8)
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

point_grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=point_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

fit = al.FitPointDataset(dataset=dataset, tracer=tracer, solver=solver)

"""
__Fit Figures__

The fit's positions component contains the residuals between the observed and model multiple-image
positions, which are irregular arrays inspected by printing.
"""
print(fit.positions.residual_map)
print(fit.positions.normalized_residual_map)
print(fit.positions.chi_squared_map)

"""
The fit subplot plots the observed and model positions together, the primary figure for checking a
point-source fit.
"""
aplt.subplot_fit_point(fit=fit)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), the fit
figures above are output to hard-disk automatically by the `Visualizer` attached to the `Analysis`
class:
"""
print(al.AnalysisPoint.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
outputs the maximum likelihood fit's figures to the fit's output folder, under `image/`
(e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`. The
machinery is described in full in `scripts/imaging/plot.py` — the same config-driven visualization
applies to every dataset type.
"""
