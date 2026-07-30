"""
Plots: Group
============

This example shows how to plot a group-scale `Imaging` dataset and a `FitImaging` fit, figure by figure and
via multi-panel subplots.

A group-scale lens differs from a galaxy-scale lens in that there are multiple lens galaxies contributing to
the lensing. In this example, there is a single main lens galaxy and two extra galaxies nearby whose mass
contributes significantly to the ray-tracing and must therefore be included in the model.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autolens.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the group-scale imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up a tracer of the main lens, two extra galaxies and source, and fit the dataset with a `FitImaging` object.
- **Fit Figures:** Plot the fit's model image, residuals and chi-squared maps individually.
- **Plane Images:** Plot the model image of each individual plane of the fit.
- **Fit Subplot:** Plot all fit quantities in one multi-panel subplot.
- **Visualizer:** How these figures are output automatically during a model-fit.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the group-scale strong lens dataset `simple` from .fits files, which is the dataset used to demonstrate
plotting. The group-scale dataset has a larger field of view than a typical galaxy-scale lens, because it
includes emission from multiple lens galaxies and a more extended lensing configuration.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
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
simulated values, via the `FitImaging` object. This is larger than a typical galaxy-scale lens mask
because the group-scale lens has emission spread over a wider area due to the multiple lens galaxies.

For a group-scale lens, the tracer contains multiple lens galaxies: a main lens galaxy and two extra
galaxies, whose combined deflection field ray-traces the source galaxy light.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Figures__

The fit's quantities — model image, residuals, chi-squared and more — are accessed as attributes
and plotted individually with `aplt.plot_array()`. For a group-scale lens, the model image includes
contributions from all lens galaxies (main and extra) as well as the lensed source galaxy.
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

Per-plane model images are accessed via `model_images_of_planes_list`. All lens galaxies (main and
extra) are at the same redshift and therefore share a single plane, so `model_images_of_planes_list[0]`
is their combined light, while `model_images_of_planes_list[1]` is the lensed source's plane.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[0], title="Plane 0 Model Image (Lens Galaxies)"
)
aplt.plot_array(
    array=fit.model_images_of_planes_list[1], title="Plane 1 Model Image (Source)"
)

"""
__Fit Subplot__

A multi-panel fit subplot is produced with `aplt.subplot_fit_imaging()`, combining the data, model
image, residual-map and chi-squared map in one figure.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Visualizer__

Group-scale fits use the same `AnalysisImaging` / `VisualizerImaging` machinery as galaxy-scale fits —
see `scripts/imaging/plot.py` for the full description of the `Visualizer` and the `plots.yaml` config
file that controls which figures it outputs. During a model-fit, all of the figures above are output
to hard-disk automatically via:
"""
print(al.AnalysisImaging.Visualizer)
