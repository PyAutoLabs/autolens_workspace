"""
Plots: Cluster
==============

This example shows how to plot a cluster-scale strong lens dataset — the multiple-image positions
of its lensed sources, individually and overlaid together on an image of the cluster.

Cluster datasets are point-source datasets: each lensed source contributes the (y,x) positions of
its multiple images (see `scripts/cluster/start_here.py` for the full data model). Plotting
therefore combines the point-source figures of `scripts/point_source/plot.py` with cluster-specific
overlay functions that show every system at once.

For an introduction to the plotting API refer to `guides/plot/start_here.py`. The final section
documents how figures are output automatically during a cluster model-fit.

__Contents__

- **Dataset:** Load the Abell 2744 point datasets and (if present) the HST image of the cluster.
- **Dataset Figures:** Inspect each system's info and plot its multiple-image positions.
- **Positions Overlay:** Plot every system's positions together, overlaid on the cluster image.
- **Image Group Zooms:** Plot a zoomed cutout around each system's multiple images.
- **Cluster Subplot:** Plot the combined cluster dataset subplot.
- **Tracer:** Build the truth-model tracer from the dataset's mass CSVs.
- **Critical Curves:** Plot the multi-plane critical curves of the cluster.
- **Caustics:** Plot the multi-plane caustics of the cluster.
- **Visualizer:** How figures are output automatically during a cluster model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the real Abell 2744 dataset, whose multiple-image positions ship with the workspace as
spreadsheet-editable CSV files.

The HST H-band image of the cluster is used for visualization only; it is downloaded on the first
run of `scripts/cluster/start_here.py`. If it is not present, the figures below are simply plotted
without a background image.
"""
dataset_name = "a2744"
dataset_path = Path("dataset") / "cluster" / dataset_name

data_fits_path = dataset_path / "data.fits"

data = None
pixel_scales = 0.3  # hips2fits returns 0.3"/pixel for this cutout.

if data_fits_path.exists():
    data = al.Array2D.from_fits(file_path=data_fits_path, pixel_scales=pixel_scales)

    aplt.plot_array(array=data, title="Abell 2744")

"""
The per-source point datasets are loaded from a single CSV. `al.list_from_csv` returns a
`List[PointDataset]` where each entry carries the source's `positions`, `positions_noise_map` and
`redshift`.
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

"""
__Dataset Figures__

Each system's `info` summarizes its name, positions, noise-map values and redshift, and its
multiple-image positions are plotted with `aplt.plot_grid()` — exactly as for a single
point-source dataset (see `scripts/point_source/plot.py`).
"""
for dataset in dataset_list:
    print(dataset.info)
    print(f"Redshift: {dataset.redshift}")

for dataset in dataset_list:
    aplt.plot_grid(
        grid=al.Grid2DIrregular(np.atleast_2d(dataset.positions)),
        title=dataset.name,
    )

"""
__Positions Overlay__

The figure that matters most for a cluster is all systems together: `aplt.plot_positions_overlay()`
plots every source's multiple-image positions in one figure, each system in its own color, overlaid
on the cluster image when one is available.
"""
positions_list = [dataset.positions for dataset in dataset_list]

aplt.plot_positions_overlay(
    positions_list=positions_list,
    image=data,
    pixel_scales=pixel_scales,
)

"""
__Image Group Zooms__

The overlay above shows the whole cluster, where individual multiple images are only a few pixels
across. `aplt.plot_image_group_zooms()` instead plots one zoomed cutout per system, so each set of
multiple images can be inspected individually.

 - `zoom_arcsec`: the width of each cutout.
 - `max_cols`: how many cutouts are placed per row.

It requires the cluster image, so it is skipped when the HST cutout has not been downloaded.
"""
if data is not None:
    aplt.plot_image_group_zooms(
        positions_list=positions_list,
        image=data,
        pixel_scales=pixel_scales,
        zoom_arcsec=6.0,
    )

"""
__Cluster Subplot__

The `aplt.subplot_cluster_dataset()` function combines the cluster dataset's figures into one
multi-panel subplot.
"""
aplt.subplot_cluster_dataset(
    positions_list=positions_list,
    image=data,
    pixel_scales=pixel_scales,
)

"""
__Tracer__

The figures above show the data alone. The remaining figures show the lens model, which requires a
`Tracer`.

The dataset ships with its mass model in `mass.csv` and `point.csv`. `al.galaxy_af_models_from_csv_tables()`
turns these into `af.Model[Galaxy]` objects whose parameters are all fixed at the CSV values
(`prior_count` is 0), so `instance_from_prior_medians()` returns the exact truth galaxies rather
than a sample.
"""
mass_table = al.galaxy_models_from_csv(file_path=dataset_path / "mass.csv", family="mass")
point_table = al.galaxy_models_from_csv(file_path=dataset_path / "point.csv", family="point")

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

galaxies = [model.instance_from_prior_medians() for model in galaxy_models.values()]

tracer = al.Tracer(galaxies=galaxies)

grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=1.0)

"""
A cluster is a multi-plane system: one lens plane and one source plane per lensed system. The
critical curves and caustics differ for every source plane, so a plane is chosen by index.
"""
plane_redshifts = [float(plane.redshift) for plane in tracer.planes]

print(f"Plane redshifts: {plane_redshifts}")

plane_indices = [1, len(plane_redshifts) - 1]

"""
__Critical Curves__

`aplt.plot_critical_curves()` plots the tangential critical curves — the loci of maximum
magnification in the image-plane — for the chosen source planes, overlaid on the cluster image.

 - `include_radial`: also plot the radial critical curves.
"""
aplt.plot_critical_curves(
    tracer,
    grid=grid,
    image=data,
    pixel_scales=pixel_scales,
    plane_indices=plane_indices,
)

"""
__Caustics__

`aplt.plot_caustics()` plots the source-plane counterpart of the critical curves. A source inside a
caustic is multiply imaged, so the caustics show where a source must lie to produce the multiple
images plotted above.
"""
aplt.plot_caustics(
    tracer,
    grid=grid,
    plane_indices=plane_indices,
)

"""
__Visualizer__

A cluster model-fit (see `scripts/cluster/start_here.py`) fits every system's positions
simultaneously, with one point-source analysis per system combined into a factor graph. During the
fit, visualization is therefore performed by the point-source `Visualizer` attached to each
analysis:
"""
print(al.AnalysisPoint.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, each system's
maximum likelihood fit figures are output to the fit's output folder, under `image/`.

Which figures are output is controlled by the config file `config/visualize/plots.yaml`. The
machinery is described in full in `scripts/imaging/plot.py` — the same config-driven visualization
applies to every dataset type.
"""
