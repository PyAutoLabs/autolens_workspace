"""
Plots: Double Einstein Ring
===========================

This example illustrates the plotting API for double Einstein ring systems, which have
more than two planes at different redshifts.

The new API uses:

 - `aplt.plot_array()` — plot any 2D array.
 - `aplt.subplot_fit_imaging()` — multi-panel fit overview.

For pixelized source reconstructions, inversion quantities are accessed via `fit.inversion`
and plotted with `aplt.plot_array()`.

__Start Here Notebook__

Refer to `plots/start_here.ipynb` for an introduction to the new plotting API.

__Contents__

**Setup:** General setup for the analysis.
**Fit Imaging:** Plot individual fit attributes with `aplt.plot_array()`.
**Full Subplot:** A multi-panel subplot overview is produced with `aplt.subplot_fit_imaging()`.
**Pixelized Source Reconstruction:** Now set up a double Einstein ring fit using pixelized source reconstructions.
**Inversion:** The inversion is computed directly from a `Tracer` using `TracerToInversion`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Setup__

Set up the double Einstein ring dataset and fit.
"""
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/advanced/double_einstein_ring/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
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
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.15, -0.15), intensity=1.2, effective_radius=0.1
    ),
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.45, 0.45), intensity=0.6, effective_radius=0.07
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Imaging__

Plot individual fit attributes with `aplt.plot_array()`.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.noise_map, title="Noise Map")
aplt.plot_array(array=fit.signal_to_noise_map, title="Signal-to-Noise Map")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
__Per-Plane Images__

For a double Einstein ring (3-plane system), per-plane images are accessed via
`model_images_of_planes_list`, which has one entry per plane.
"""
aplt.plot_array(array=fit.model_images_of_planes_list[0], title="Plane 0 Model Image")
aplt.plot_array(array=fit.model_images_of_planes_list[1], title="Plane 1 Model Image")
aplt.plot_array(array=fit.model_images_of_planes_list[2], title="Plane 2 Model Image")

"""
__Full Subplot__

A multi-panel subplot overview is produced with `aplt.subplot_fit_imaging()`.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Pixelized Source Reconstruction__

Now set up a double Einstein ring fit using pixelized source reconstructions.
"""
source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        mesh=al.mesh.RectangularAdaptDensity(shape=(24, 24)),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    pixelization=al.Pixelization(
        mesh=al.mesh.RectangularAdaptDensity(shape=(24, 24)),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
Plot the fit overview.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
The pixelized source reconstructions are accessed via the `inversion` property of the fit.

For a double Einstein ring there are two reconstructions (one per source plane), indexed
by their position in `fit.inversion.reconstruction_dict`.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Plane 1 Model Image (Pixelized)",
)
aplt.plot_array(
    array=fit.model_images_of_planes_list[2],
    title="Plane 2 Model Image (Pixelized)",
)

"""
__Inversion__

The inversion is computed directly from a `Tracer` using `TracerToInversion`.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
Plot the reconstructed source for each pixelization index.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Inversion Reconstruction (Plane 1)",
)
aplt.plot_array(
    array=fit.model_images_of_planes_list[2],
    title="Inversion Reconstruction (Plane 2)",
)

"""
Finish.
"""
