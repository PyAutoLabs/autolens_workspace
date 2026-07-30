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
- **Cluster Subplot:** Plot the combined cluster dataset subplot.
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
