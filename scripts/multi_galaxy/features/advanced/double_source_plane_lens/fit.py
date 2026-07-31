"""
Fits: DSPL (Multi Galaxy)
=========================

This script fits a multi-galaxy double source-plane lens without a non-linear search, so the multi-plane tracer
and the objects it produces can be inspected directly.

__Contents__

- **Dataset:** Load the multi-galaxy DSPL dataset that is fitted.
- **Mask:** Standard set up of the mask that is fitted.
- **Centres:** The centres of the co-dominant deflectors.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **MGE Bases:** One basis per deflector, and one per source.
- **Galaxies:** Compose the three planes.
- **Tracer:** Build the multi-plane tracer.
- **Fit:** Fit the dataset and solve every basis's intensities.
- **Multi-Plane Ray-Tracing:** Confirm the deflection chain plane by plane.
- **Basis Images:** Plot each galaxy's basis with its solved amplitudes.
- **Wrap Up:** Where to go next.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' DSPL where:

 - Each co-dominant deflector's light is an MGE and its mass an `Isothermal`, at their simulated values.
 - The system has a single overall `ExternalShear` at the system centre.
 - `source_0` at z=1.0 has an MGE and an `IsothermalSph` mass, since it deflects `source_1`.
 - `source_1` at z=2.0 has an MGE only.

__What Is Different Here__

`multi_galaxy/fit.py` builds a two-plane tracer whose deflection field is the sum of both deflectors'. This one
has three planes, and the chain accumulates: light reaching the image plane from z=2.0 has been deflected by
both deflectors at z=0.5 **and** by `source_0`'s mass at z=1.0.

The `__Multi-Plane Ray-Tracing__` section below reads that chain out one plane at a time.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/fit.py` for the multi-galaxy fit anatomy and
`imaging/features/advanced/double_source_plane_lens/fit.py` for the single-deflector DSPL.
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

The `dspl` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "dspl"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/double_source_plane_lens/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Mask__

The standard 3.0" circular mask, sized to contain both Einstein rings.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Over Sampling__
"""
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__MGE Bases__

An MGE basis per galaxy, built by hand as a `Basis` of linear `Gaussian` profiles with log-spaced widths.
`al.model_util.mge_model_from` is the helper the modeling scripts use, but it returns an `af.Model`; a concrete
fit needs concrete profiles.

Every Gaussian's intensity is solved by linear algebra, so none is set here.

The sources get narrower `sigma` ranges than the deflectors: they are compact, and Gaussians much wider than the
source only add ways to fit noise.
"""
total_gaussians = 20

log10_sigma_lens = np.linspace(-2, np.log10(mask_radius), total_gaussians)
log10_sigma_source = np.linspace(-2, np.log10(0.5), total_gaussians)


def build_basis(centre, log10_sigma_list):
    return al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Gaussian(
                centre=centre,
                ell_comps=(0.0, 0.0),
                sigma=10 ** log10_sigma_list[i],
            )
            for i in range(total_gaussians)
        ]
    )


lens_bulge_list = [
    build_basis((centre[0], centre[1]), log10_sigma_lens)
    for centre in main_lens_centres
]

source_0_bulge = build_basis((0.0, 0.03), log10_sigma_source)

source_1_bulge = build_basis((-0.25, 0.28), log10_sigma_source)

"""
__Galaxies__

The three planes, at the values `simulator.py` used.

`source_0` carries mass as well as light, which is what makes the tracing multi-plane rather than two separate
single-plane problems.
"""
lens_masses = [
    al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
    al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
]

lens_galaxies = [
    al.Galaxy(redshift=0.5, bulge=bulge, mass=mass)
    for bulge, mass in zip(lens_bulge_list, lens_masses)
]

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_0 = al.Galaxy(
    redshift=1.0,
    bulge=source_0_bulge,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.03), einstein_radius=0.15),
)

source_1 = al.Galaxy(
    redshift=2.0,
    bulge=source_1_bulge,
)

"""
__Tracer__

PyAutoLens orders galaxies by redshift internally, so passing them in any order produces the same three-plane
chain: z=0.5, z=1.0, z=2.0.
"""
tracer = al.Tracer(galaxies=lens_galaxies + [shear_galaxy, source_0, source_1])

"""
__Fit__

`FitImaging` solves every basis's Gaussian intensities by linear algebra and computes the model image, residuals
and log likelihood.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood = {fit.log_likelihood}")

"""
__Multi-Plane Ray-Tracing__

Read the deflection chain out one plane at a time. `traced_grid_2d_list_from` returns one grid per plane, each
being the image-plane grid after the deflections of every plane before it have been applied.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.grid)

print(f"Number of planes traced through: {len(traced_grids)}")
print(f"Plane 0 (image plane)        — first coordinate: {traced_grids[0][0]}")
print(f"Plane 1 (source_0 at z=1.0)  — first coordinate: {traced_grids[1][0]}")
print(f"Plane 2 (source_1 at z=2.0)  — first coordinate: {traced_grids[2][0]}")

"""
The difference between plane 1 and plane 2 is where the multi-plane structure shows up. If `source_0` had no
mass, plane 2 would be plane 1 scaled by a single geometric factor. It does have mass, so the extra deflection
varies across the grid instead.
"""
print(
    f"Plane 1 -> 2 displacement at the first coordinate: "
    f"{traced_grids[2][0] - traced_grids[1][0]}"
)

"""
__Basis Images__

After the fit, every linear Gaussian has a solved intensity. The fit exposes a tracer with those profiles
converted to ordinary light profiles, which is what the basis plots need.
"""
tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

n_lens = len(lens_galaxies)

for i in range(n_lens):
    subplot_basis_image(basis=tracer_fitted.galaxies[i].bulge, grid=plot_grid)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/double_source_plane_lens/modeling.py` — fitting this model with a search.
 - `multi_galaxy/features/advanced/double_source_plane_lens/likelihood_function.py` — the same fit broken into
   its individual steps.
 - `multi_galaxy/fit.py` — the two-plane multi-galaxy fit this one extends.
 - `imaging/features/advanced/double_source_plane_lens/fit.py` — the single-deflector DSPL.
"""
