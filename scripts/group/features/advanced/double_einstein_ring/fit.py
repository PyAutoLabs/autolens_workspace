"""
Features: Group Double Einstein Ring Fit
========================================

A group-scale double Einstein ring lens, where two source galaxies at different redshifts are lensed by multiple
main lens galaxies at the lens-plane redshift.

This script demonstrates the API for fitting a group-scale double Einstein ring system via the standard `Tracer`
and `FitImaging` objects, without invoking a non-linear search. The two main lens galaxies are composed using
the group `lens_dict` convention (loaded from a JSON file of centres), and the two source galaxies share the
3-plane ray-tracing structure used in the imaging double Einstein ring example.

The source galaxies are both modelled with a Multi Gaussian Expansion (MGE), as is each main lens galaxy's bulge.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset, Mask, Over Sampling:** Standard set up.
- **Main Lens Centres:** Load the two main lens galaxy centres from JSON.
- **MGE Bases:** Build linear-Gaussian bases for each main lens galaxy bulge and for each source galaxy bulge.
- **Galaxies:** Compose `lens_dict` (two main lens galaxies, each with MGE bulge + mass), `source_0` (MGE bulge
  + mass), `source_1` (MGE bulge).
- **Tracer:** Build the three-plane `Tracer` from `list(lens_dict.values()) + [source_0, source_1]`.
- **Fit:** Run `FitImaging` and inspect the fit.
- **Multi-Plane Ray-Tracing:** A short tour confirming the deflection chain accumulates contributions from both
  main lens galaxies and from `source_0`'s mass.
- **Wrap Up.**

__Prerequisites__

This script combines the group-scale `lens_dict` API and the double Einstein ring multi-plane API. Read first:

 - `autolens_workspace/scripts/group/fit.py` — the standard group-scale fit.
 - `autolens_workspace/scripts/imaging/features/advanced/double_einstein_ring/fit.py` — the single-lens double
   Einstein ring fit, including the MGE basis pattern used here.
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

Load the group double Einstein ring dataset.
"""
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "group" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/double_einstein_ring/simulator.py",
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
__Main Lens Centres__

Load the two main lens galaxy centres saved by the simulator.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

A 4.0" circular mask, centred at the origin (midpoint of the two main lens galaxies), large enough to enclose
both Einstein rings.
"""
mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

Adaptive over-sampling at each main lens galaxy centre.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__MGE Bases__

We build a linear-Gaussian `Basis` for each main lens galaxy bulge and for each source galaxy bulge. The Gaussians
share log10-spaced `sigma` values and have spherical symmetry for simplicity. The `intensity` of each Gaussian
is solved for at fit time via linear algebra.
"""
total_gaussians = 30
log10_sigma_list_lens = np.linspace(-2, np.log10(2.0), total_gaussians)
log10_sigma_list_source = np.linspace(-2, np.log10(0.5), total_gaussians)


def build_basis(centre, log10_sigma_list):
    gaussian_list = [
        al.lp_linear.Gaussian(
            centre=centre,
            ell_comps=(0.0, 0.0),
            sigma=10 ** log10_sigma_list[i],
        )
        for i in range(total_gaussians)
    ]
    return al.lp_basis.Basis(profile_list=gaussian_list)


"""
__Galaxies__

The lens-plane (z=0.5) is composed via the group `lens_dict` convention. Each main lens galaxy has an MGE bulge
centred on its loaded position, and an `IsothermalSph` mass profile matching the simulator's true values.

`source_0` at z=1.0 has an MGE bulge AND an `IsothermalSph` mass — it deflects light from `source_1`.
`source_1` at z=2.0 has an MGE bulge only.
"""
lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    lens_dict[f"lens_{i}"] = al.Galaxy(
        redshift=0.5,
        bulge=build_basis(
            centre=(centre[0], centre[1]), log10_sigma_list=log10_sigma_list_lens
        ),
        mass=al.mp.IsothermalSph(centre=(centre[0], centre[1]), einstein_radius=1.2),
    )

source_0_bulge = build_basis(
    centre=(0.0, 0.0), log10_sigma_list=log10_sigma_list_source
)
source_1_bulge = build_basis(
    centre=(-0.3, 0.3), log10_sigma_list=log10_sigma_list_source
)

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=source_0_bulge,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.25),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=source_1_bulge,
)

"""
__Tracer__

Build the multi-plane Tracer. Galaxies are ordered internally by redshift, so the deflection chain runs:

  Plane 0 (image)            : theta
  Plane 1 (source_0, z=1.0)  : theta - sum_over_main_lens_galaxies alpha_lens_i(theta)
  Plane 2 (source_1, z=2.0)  : theta - sum_over_main_lens_galaxies alpha_lens_i(theta)
                                     - beta_01 * alpha_source_0(plane_1_grid)

Note the contribution to `source_1`'s grid from EACH main lens galaxy as well as from `source_0`'s mass.
"""
tracer = al.Tracer(galaxies=list(lens_dict.values()) + [source_0, source_1])

"""
__Fit__

Run `FitImaging` to solve for every Gaussian's `intensity` via linear algebra and to compute the model image,
residuals and log likelihood.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Multi-Plane Ray-Tracing__

Confirm the deflection chain by extracting one grid per plane.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.grid)

print(f"Number of planes traced through: {len(traced_grids)}")
print(f"Plane 0 (image-plane)        — first coord: {traced_grids[0][0]}")
print(f"Plane 1 (source_0 at z=1.0)  — first coord: {traced_grids[1][0]}")
print(f"Plane 2 (source_1 at z=2.0)  — first coord: {traced_grids[2][0]}")

"""
After the fit, the linear MGE Gaussians have been assigned intensities. The fit exposes a Tracer with the
linear profiles converted to standard light profiles via
`fit.model_obj_linear_light_profiles_to_light_profiles`, which we use to visualise each MGE basis with its
solved-for amplitudes.
"""
tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

# Galaxies are ordered by redshift, so the first len(lens_dict) entries are the main lens galaxies,
# followed by source_0 (index N) and source_1 (index N+1).
n_lens = len(lens_dict)

for i in range(n_lens):
    subplot_basis_image(basis=tracer_fitted.galaxies[i].bulge, grid=plot_grid)

subplot_basis_image(basis=tracer_fitted.galaxies[n_lens].bulge, grid=plot_grid)
subplot_basis_image(basis=tracer_fitted.galaxies[n_lens + 1].bulge, grid=plot_grid)

print(f"\nFit log_likelihood: {fit.log_likelihood}")

"""
__Wrap Up__

This script demonstrated the API for fitting a group-scale double Einstein ring system without invoking a
non-linear search. The two new aspects relative to the imaging double Einstein ring example are:

 1. The lens-plane is composed via the group `lens_dict` convention, scaling naturally to any number of main
    lens galaxies.
 2. The deflection chain to `source_1`'s plane accumulates contributions from EVERY main lens galaxy at z=0.5
    plus `source_0`'s mass.

For a realistic group DSPL fit on real data, use `chaining.py` (two chained searches) or `slam.py` (full
SLaM pipeline). `modeling.py` provides a tutorial single-search example, but cheats by initialising priors at
the true simulator values and is not appropriate for real data.
"""
