"""
Plots: Imaging
==============

This example shows how to plot an `Imaging` dataset and a `FitImaging` fit, figure by figure and
via multi-panel subplots.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autolens.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up a tracer and fit the dataset with a `FitImaging` object.
- **Fit Figures:** Plot the fit's model image, residuals and chi-squared maps individually.
- **Plane Images:** Plot the model image of each individual plane of the fit.
- **Plane Subplots:** Plot a multi-panel subplot for each plane of the fit.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Fit Subplot (Log10):** The same subplot on a log10 colour scale.
- **Tracer Subplot:** Plot the fit's tracer quantities (convergence, potential, deflections).
- **Outputting to FITS:** Write the dataset to a FITS file instead of an image.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens dataset `simple` from .fits files, which is the dataset used to demonstrate
plotting.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
__Dataset Figures__

The dataset's data, noise-map and PSF are attributes, each plotted individually with
`aplt.plot_array()`.
"""
aplt.plot_array(array=dataset.data, title="Data")
aplt.plot_array(array=dataset.noise_map, title="Noise Map")
aplt.plot_array(array=dataset.psf.kernel, title="PSF")

"""
__Dataset Subplot__

A multi-panel subplot of the dataset is produced with `aplt.subplot_imaging_dataset()`, including
the data, noise-map, PSF and signal-to-noise map.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Fit__

To plot a fit, we mask the dataset and fit it with a tracer whose galaxies match the true
simulated values, via the `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Figures__

The fit's quantities — model image, residuals, chi-squared and more — are accessed as attributes
and plotted individually with `aplt.plot_array()`.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.noise_map, title="Noise Map")
aplt.plot_array(array=fit.signal_to_noise_map, title="Signal-to-Noise Map")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
__Plane Images__

Per-plane model images are accessed via `model_images_of_planes_list`, for example to inspect the
lens galaxy's light (plane 0) and the lensed source (plane 1) separately.
"""
aplt.plot_array(array=fit.model_images_of_planes_list[0], title="Plane 0 Model Image")
aplt.plot_array(array=fit.model_images_of_planes_list[1], title="Plane 1 Model Image")

"""
__Plane Subplots__

`aplt.subplot_fit_imaging_of_planes()` produces a 4-panel subplot per plane (data, subtracted
image, model image and plane image), saved as one figure per plane.

Pass `plane_index` to produce the subplot for a single plane only.
"""
aplt.subplot_fit_imaging_of_planes(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit, plane_index=1)

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`, combining the data, model
image, residual-map and chi-squared map in one figure.

For a tracer with only one plane (no source plane) this function automatically switches to a
simpler 2 x 3 layout, so the same call works for single-plane and multi-plane fits alike.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Fit Subplot (Log10)__

`aplt.subplot_fit_imaging_log10()` is the same subplot with the image panels on a `log10` colour
scale, which makes faint extended emission (e.g. the outskirts of the lensed source) visible.
"""
aplt.subplot_fit_imaging_log10(fit=fit)

"""
__Tracer Subplot__

`aplt.subplot_fit_imaging_tracer()` plots the tracer quantities of the fit — image, convergence,
potential and deflection angles — without needing to build the tracer and grid separately.
"""
aplt.subplot_fit_imaging_tracer(fit=fit)

"""
__Outputting to FITS__

Figures are images, but the dataset itself can be written to a FITS file with
`aplt.fits_imaging()`. Passing `file_path` writes a single multi-HDU file with named extensions
(`mask`, `data`, `psf`, `noise_map`); passing `data_path` / `psf_path` / `noise_map_path` instead
writes each component to its own file.
"""
output_path = Path("output") / "plot" / "imaging"
output_path.mkdir(parents=True, exist_ok=True)

aplt.fits_imaging(
    dataset=dataset,
    file_path=output_path / "dataset.fits",
    overwrite=True,
)

"""
__Visualizer__

During a model-fit (e.g. `search.fit(model=model, analysis=analysis)` in `modeling.py`), all of
the figures above are output to hard-disk automatically — you do not need to call the plotting
functions yourself to inspect a fit's progress or results.

This is performed by the `Visualizer` attached to the `Analysis` class:
"""
print(al.AnalysisImaging.Visualizer)

"""
At regular intervals during the non-linear search, and again once it finishes, the `Visualizer`
computes the maximum likelihood fit and outputs its figures to the fit's output folder, under
`image/` (e.g. `output/<path_prefix>/<name>/image/`).

Which figures are output is controlled by the config file `config/visualize/plots.yaml`, for
example:

 - `dataset` -> `subplot_dataset`: The multi-panel dataset subplot shown above.
 - `fit` -> `subplot_fit`: The multi-panel fit subplot shown above.
 - `fit` -> `subplot_of_planes`: The per-plane model images shown above.
 - `tracer` -> `subplot_tracer`: A subplot of the tracer's image, convergence and potential.

Setting an entry to `true` or `false` in `plots.yaml` therefore switches that figure on or off
for every model-fit, without changing code.

When multiple datasets are fitted simultaneously, the `Visualizer` also outputs figures combining
all datasets in one subplot — see `scripts/multi/plot.py` for details.
"""
