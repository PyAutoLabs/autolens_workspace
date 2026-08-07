"""
Plots: Multi Galaxy
===================

This example shows how to plot a multi-galaxy `Imaging` dataset and a `FitImaging` fit, figure by figure and
via multi-panel subplots.

A multi-galaxy lens has two or more galaxies of comparable mass which both contribute significantly to the
lensing of a single background source. The plotting API is **identical** to the galaxy-scale case — a fit
neither knows nor cares how many galaxies deflect the light — so everything in this example applies equally
to `imaging/plot.py`.

Quantities are computed from the dataset and fit objects via their attributes and methods, and
passed to the plotting functions in `autolens.plot` (imported as `aplt`). For an introduction to
the plotting API itself (customization, output to disk, config defaults, overlays) refer to
`guides/plot/start_here.py`.

The final section documents the `Visualizer`, which outputs all of these figures automatically
during a model-fit via the `Analysis` object.

__Contents__

- **Dataset:** Load the multi-galaxy imaging dataset used throughout this example.
- **Dataset Figures:** Plot the dataset's data, noise-map and PSF individually.
- **Dataset Subplot:** Plot all dataset quantities in one multi-panel subplot.
- **Fit:** Set up a tracer of the two co-dominant lens galaxies and source, and fit the dataset with a `FitImaging` object.
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

Load the multi-galaxy strong lens dataset `simple` from .fits files, which is the dataset used to demonstrate
plotting.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Dataset Noise Scaling__

The dataset includes a faint extra galaxy, whose light is not associated with the strong lens but blends
into the field. A `mask_extra_galaxies.fits` covering it ships with the dataset, and its pixels are scaled
to zero data and very large noise so they contribute negligibly to any fit, before the dataset's own
quantities are plotted below.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,  # `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

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

For a multi-galaxy lens, the tracer contains two co-dominant lens galaxies (`lens_0` and `lens_1`) whose
deflection fields are summed, plus a separate galaxy holding the external shear (a property of the system
as a whole, not of either deflector), and the source galaxy.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Figures__

The fit's quantities — model image, residuals, chi-squared and more — are accessed as attributes
and plotted individually with `aplt.plot_array()`. For a multi-galaxy lens, the model image includes
the combined contributions of both lens galaxies as well as the lensed source galaxy.
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

Per-plane model images are accessed via `model_images_of_planes_list`. A "plane" is a *redshift*, not a
galaxy: both `lens_0` and `lens_1` are at z=0.5, so there are only two planes for the three lens-redshift
galaxies, and `model_images_of_planes_list[0]` is their **combined** light, while
`model_images_of_planes_list[1]` is the lensed source's plane.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[0],
    title="Plane 0 Model Image (Lens Galaxies)",
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

Multi-galaxy fits use the same `AnalysisImaging` / `VisualizerImaging` machinery as galaxy-scale fits —
see `scripts/imaging/plot.py` for the full description of the `Visualizer` and the `plots.yaml` config
file that controls which figures it outputs. During a model-fit, all of the figures above are output
to hard-disk automatically via:
"""
print(al.AnalysisImaging.Visualizer)
