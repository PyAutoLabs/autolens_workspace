"""
Plots: Cluster
==============

This example shows how to plot a cluster-scale strong lens dataset — the multiple-image positions
of its lensed sources, individually and overlaid on a cluster image. It then follows one family from its observed images
to the source-plane fit.

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
- **Tracer:** Build a tracer from the main lens galaxies, host halo and source redshifts.
- **Critical Curves:** Plot the multi-plane critical curves of the cluster.
- **Caustics:** Plot the multi-plane caustics of the cluster.
- **Family Fit:** Inspect the sky overview, contribution ranking, one numbered image and the common source plane, then combine them.
- **Visualizer:** How figures are output automatically during a cluster model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import sys
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

if data_fits_path.exists():
    sys.path.insert(0, str(dataset_path.resolve()))
    from pandora_image import image_from_fits

    data = image_from_fits(data_fits_path)

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
)

"""
__Tracer__

The figures above show the data alone. The remaining figures show the lens model, which requires a
`Tracer`.

For the critical curves and caustics, use the two main lens galaxies and the host halo from
`mass.csv`. Their parameters are fixed at the CSV values. Each observed family supplies a source
redshift and a named `PointSolved` profile. The family fits below will solve for its centre.
"""
mass_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "mass.csv", family="mass"
)
mass_models = al.galaxy_af_models_from_csv_tables(mass_table)
main_lenses = [model.instance_from_prior_medians() for model in mass_models.values()]
sources = [
    al.Galaxy(redshift=dataset.redshift, **{dataset.name: al.ps.PointSolved()})
    for dataset in dataset_list
]
tracer = al.Tracer(galaxies=main_lenses + sources)

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
__Family Fit__

Follow the four observed images of Pandora's `point_6`. To reproduce the gallery's fixed starting
model, add the 188 cluster members from the luminosity catalogue. Their velocity dispersions and
cut radii follow the scaling relation in `scripts/cluster/start_here.py`, with the reference
dispersion fixed at its prior midpoint of 150 km/s. These plots diagnose this starting model.
"""
members = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
source_redshift = max(dataset.redshift for dataset in dataset_list)
member_galaxies = []
for centre, luminosity in zip(members.centres, members.luminosities, strict=True):
    mass = al.mp.dPIEMassSph(
        centre=tuple(centre),
        sigma=150 * luminosity**0.25,
        r_cut=5 * luminosity**0.7,
        r_core=0.0,
        redshift_object=0.308,
        redshift_source=source_redshift,
        H0=67.66,
        Om0=0.30966,
    )
    member_galaxies.append(al.Galaxy(redshift=0.308, mass=mass))

fit_tracer = al.Tracer(galaxies=main_lenses + member_galaxies + sources)
fits = [
    al.FitPointDataset(
        dataset=dataset,
        tracer=fit_tracer,
        solver=None,
        fit_positions_cls=al.FitPositionsSourceSolved,
    )
    for dataset in dataset_list
]
fit = next(fit for fit in fits if fit.dataset.name == "point_6")

"""
**Find the family on the sky.** The overview numbers the observed images of `point_6`. Passing
`fits=fits` shows the other six families for context and keeps colours consistent across panels.
Each function also works with just the selected fit.
"""
aplt.plot_family_overview(fit=fit, fits=fits, image=data)

"""
**Read the ranking.** Each bar is one image's contribution to the positions statistic. Image 4
contributes the most for this starting model. The axis follows the likelihood used by the fit:
for example, an all-pairs mixture reports contributions to −2 ln L.
"""
aplt.plot_family_contributions(fit=fit, fits=fits)

"""
**Inspect Image 4.** Pass the number printed beside the image. For this source-plane fit the zoom
compares the back-traced position and its uncertainty with the solved source centre. For an
image-plane fit the same call shows the observed position and the model prediction. Omit
`image_number` to see the whole family in a shared detail panel.
"""
aplt.plot_family_image_detail(fit=fit, fits=fits, image_number=4)

"""
**Compare all back-traced images.** Every numbered image is traced to the same source plane.
The plus marks the fitted centre. This panel is also available for an image-plane fit.
"""
aplt.plot_family_source_plane(fit=fit, fits=fits)

"""
**Put the panels together.** The subplot uses the same panels and image numbers. It shows up
to six individual zooms per page. Set `single_panel=True` to share one detail panel instead.
To arrange individual panels yourself, pass a Matplotlib `ax` to each `plot_family_*` call.
To save one directly, pass `output_path`, `output_filename` and `output_format="png"`.
"""
aplt.subplot_fit_family(fit=fit, fits=fits, image=data)

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
