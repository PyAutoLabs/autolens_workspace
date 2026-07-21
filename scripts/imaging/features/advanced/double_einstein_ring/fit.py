"""
Features: Double Einstein Ring Fit
==================================

A double Einstein ring lens is a strong lens system where two source galaxies at different redshifts are lensed
by the foreground lens galaxy. They appear as two distinct Einstein rings in the image-plane.

This script illustrates the API for performing a fit to a double Einstein ring lens via the standard `Tracer`
and `FitImaging` objects, without invoking a non-linear search. It is intended to make the multi-plane
ray-tracing API concrete before the reader moves on to `modeling.py` (search-based) or `chaining.py` / `slam.py`
(realistic, robust modeling).

The source galaxies are both modelled with a Multi Gaussian Expansion (MGE), the same source parameterization
used in `chaining.py` and `slam.py`. The MGE is built from a `Basis` of linear `Gaussian` light profiles, whose
`intensity` values are solved for via linear algebra at fit time.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **MGE Basis:** Build a `Basis` of linear Gaussians, used for both source galaxies.
- **Galaxies:** Compose the lens galaxy plus two source galaxies at different redshifts.
- **Tracer:** Build the three-plane `Tracer` that performs the multi-plane ray-tracing.
- **Fit:** Create a `FitImaging` and inspect the fit.
- **Multi-Plane Ray-Tracing:** A short tour of how the second source-plane sees deflection from both the lens
  galaxy AND the first source galaxy's mass.
- **Intensities:** The solved-for linear light profile `intensity` values for each Gaussian, per source.
- **Wrap Up:** Summary and next steps.

__Prerequisites__

This script focuses on the API specific to a double Einstein ring fit. For background on the underlying single-plane
fit API and the MGE source parameterization, you should read first:

 - `autolens_workspace/scripts/imaging/fit.py` — the standard single-plane fit.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py` — the MGE fit API and `Basis` of
   linear Gaussians.

The redshifts (`lens=0.5`, `source_0=1.0`, `source_1=2.0`) match those used by the simulator and modeling examples.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

"""
__Dataset__

Load and plot the strong lens dataset `double_einstein_ring` via .fits files.

This dataset has a double Einstein ring, due to the two source galaxies at different redshifts behind the lens galaxy.
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

Define a 3.0" circular mask, which includes both source-galaxy Einstein rings.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Apply adaptive over sampling, with finer sub-pixelization at the centre where the lens galaxy's mass is most
strongly deflecting light.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__MGE Basis__

We build a single `Basis` of linear Gaussians which we will use as the source-galaxy light model for both
`source_0` and `source_1`.

The Gaussians share a common centre and elliptical components (set to spherical here for simplicity), and have
`sigma` values spaced in log10 increments from 0.01" up to a reasonable size cap. The `intensity` of each
Gaussian is a linear parameter, solved for by linear algebra at fit time — no non-linear search is required.

Two `Basis` objects are constructed, with different sphericity centres, so that `source_0` and `source_1` each
have their own light component centred on the (known) position from the simulator.

For background on the MGE `Basis` API, see
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py`.
"""
total_gaussians = 30
log10_sigma_list = np.linspace(-2, np.log10(0.5), total_gaussians)


def build_source_basis(centre):
    gaussian_list = [
        al.lp_linear.Gaussian(
            centre=centre,
            ell_comps=(0.0, 0.0),
            sigma=10 ** log10_sigma_list[i],
        )
        for i in range(total_gaussians)
    ]
    return al.lp_basis.Basis(profile_list=gaussian_list)


# Centres match the simulator: source_0 is offset from the lens, source_1 is on the far side of the lens.

source_0_bulge = build_source_basis(centre=(-0.15, -0.15))
source_1_bulge = build_source_basis(centre=(-0.45, 0.45))

"""
The Gaussians of each basis cannot be plotted yet because their `intensity` values have not been solved for —
linear light profiles only acquire an `intensity` once a `FitImaging` runs its linear algebra step. After the
fit below, we visualise each source's MGE basis with its solved-for intensities.

We set up the plotting grid we will use post-fit.
"""
plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

We now compose the three galaxies that form the double Einstein ring system:

 - `lens` (z=0.5): an `Isothermal` mass profile, matching the simulator. The lens galaxy has no light in this
   simulated dataset.
 - `source_0` (z=1.0): the MGE basis above as a light component, AND an `IsothermalSph` mass profile. `source_0`
   acts as both a light source AND a deflector — its mass distribution bends the light from the higher-redshift
   `source_1`, contributing to the second Einstein ring.
 - `source_1` (z=2.0): only an MGE light component. It contributes light, but its own mass is negligible at
   this redshift in the simulated data.

The mass profile values for `lens` and `source_0` are set to the simulator's true values, so the fit visibly
recovers both Einstein rings without a search.
"""
lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=source_0_bulge,
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=source_1_bulge,
)

"""
__Tracer__

The `Tracer` performs the multi-plane ray-tracing. PyAutoLens orders the galaxies internally by redshift, so
the tracer first deflects image-plane (y,x) coordinates through the lens-plane (z=0.5) onto `source_0`'s plane
(z=1.0), then continues to deflect through `source_0`'s mass to reach `source_1`'s plane (z=2.0).

Both the lens galaxy's mass AND `source_0`'s mass contribute to the deflection map applied to coordinates
arriving at `source_1`'s plane. This is what makes double Einstein ring systems sensitive to angular diameter
distance ratios, and therefore to cosmological parameters.
"""
tracer = al.Tracer(galaxies=[lens, source_0, source_1])

"""
__Fit__

We pass the `Tracer` to a `FitImaging` to fit the dataset. The fit performs the multi-plane ray-tracing, evaluates
each source galaxy's light at its own source-plane, sums the resulting image-plane contributions, convolves with
the PSF, and computes the residuals against the data.

The `linear_light_profile_intensity_dict` of the fit will hold a solved-for `intensity` for every Gaussian in
both source MGE bases.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Multi-Plane Ray-Tracing__

The tracer exposes per-plane ray-traced grids via `traced_grid_2d_list_from`. The list returned has one grid per
plane (image-plane, `source_0`-plane, `source_1`-plane).

Inspecting these grids confirms the chained deflection: the grid arriving at `source_1`'s plane has been
deflected first by the lens galaxy and then by `source_0`'s mass.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.grid)

print(f"Number of planes traced through: {len(traced_grids)}")
print(f"Plane 0 (image-plane)        — first 3 coordinates: {traced_grids[0][:3]}")
print(f"Plane 1 (source_0 at z=1.0)  — first 3 coordinates: {traced_grids[1][:3]}")
print(f"Plane 2 (source_1 at z=2.0)  — first 3 coordinates: {traced_grids[2][:3]}")

"""
__Intensities__

After the fit, every linear Gaussian in each source's MGE basis has been assigned an `intensity` via linear
algebra. These are available via the fit's `linear_light_profile_intensity_dict`, keyed by light profile object.

We print the intensity of the first Gaussian in each source's basis to confirm both sources have been
reconstructed.
"""
print(
    f"\nFirst Gaussian intensity, source_0 = "
    f"{fit.linear_light_profile_intensity_dict[source_0_bulge.profile_list[0]]}"
)
print(
    f"First Gaussian intensity, source_1 = "
    f"{fit.linear_light_profile_intensity_dict[source_1_bulge.profile_list[0]]}"
)

"""
A `Tracer` where every linear light profile has been replaced with an ordinary light profile carrying its
solved-for `intensity` is also accessible from the fit, which is useful for visualising each MGE basis with
its actual reconstructed amplitude.
"""
tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

subplot_basis_image(basis=tracer_fitted.galaxies[1].bulge, grid=plot_grid)
subplot_basis_image(basis=tracer_fitted.galaxies[2].bulge, grid=plot_grid)

"""
__Wrap Up__

This script demonstrated the multi-plane ray-tracing and MGE source API for a double Einstein ring lens, without
invoking a non-linear search.

In a real modeling workflow:

 - `modeling.py` shows how to fit the same system using `Nautilus`, but "cheats" by initialising priors at the
   true values. It is therefore only useful as a tutorial.
 - `chaining.py` is the practical workflow — two chained searches that initialise the lens and `source_0` first,
   then introduce `source_1`. This is the script you'll actually use to fit data.
 - `slam.py` is the most robust pipeline for production-quality DSPL modeling, ending in a pixelized source
   reconstruction.

The key takeaway from this script is that double Einstein rings are fit with the same `Tracer` + `FitImaging`
objects as any other lens; the only difference is that the `Tracer` contains three (or more) galaxies at
different redshifts and `source_0` carries both light AND mass.
"""
