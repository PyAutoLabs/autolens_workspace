"""
Features: Mass Stellar Dark Fit
===============================

A decomposed mass model splits the lens galaxy's total mass into a stellar component (tied to its observed light
via a mass-to-light ratio) and a dark matter component (typically an NFW halo). The total deflection at every
image-plane coordinate is the sum of the deflections produced by each component, plus any external shear.

This script illustrates the API for performing a fit to a decomposed-mass lens via the standard `Tracer` and
`FitImaging` objects, without invoking a non-linear search. It is intended to make the decomposed-mass
deflection composition concrete before the reader moves on to `modeling.py` (search-based) or `chaining.py` /
`slam.py` (realistic, robust modeling).

The lens galaxy uses a linear `lmp.Sersic` light-and-mass profile (a Sersic that simultaneously acts as a light
profile and a stellar mass profile, coupled by a single `mass_to_light_ratio` parameter), plus a spherical NFW
dark matter halo and an external shear. The source galaxy is modelled with a Multi Gaussian Expansion (MGE), the
same source parameterization used in `chaining.py` and `slam.py`.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **MGE Basis:** Build a `Basis` of linear Gaussians, used for the source galaxy.
- **Galaxies:** Compose the decomposed-mass lens galaxy plus the MGE source.
- **Tracer:** Build the two-plane `Tracer` that performs the ray-tracing.
- **Fit:** Create a `FitImaging` and inspect the fit.
- **Decomposed Deflection:** A short tour of how the total lens deflection is the sum of the stellar, dark and
  shear contributions, and how the same composition shows up in convergence.
- **Intensities:** The solved-for linear light profile `intensity` values for each MGE Gaussian.
- **Wrap Up:** Summary and next steps.

__Prerequisites__

This script focuses on the API specific to a decomposed-mass fit. For background on the underlying single-plane
fit API and the MGE source parameterization, you should read first:

 - `autolens_workspace/scripts/imaging/fit.py` — the standard single-plane fit.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py` — the MGE fit API and `Basis` of
   linear Gaussians.

The galaxy redshifts (`lens=0.5`, `source=1.0`), the lens `lmp.Sersic` mass-to-light ratio (0.2), and the dark
`NFWSph` parameters (`kappa_s=0.1`, `scale_radius=20.0`) match those used by the simulator and modeling examples.
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

Load and plot the strong lens dataset `mass_stellar_dark` via .fits files.

This dataset shows a single Einstein ring, simulated from a lens galaxy whose total mass is the sum of a Sersic
stellar component (with mass tied to its light via a constant mass-to-light ratio) and a spherical NFW dark
matter halo.
"""
dataset_name = "mass_stellar_dark"
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
        [
            sys.executable,
            "scripts/imaging/features/advanced/mass_stellar_dark/simulator.py",
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

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
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

Apply adaptive over sampling, with finer sub-pixelization at the centre where the lens galaxy's light and
stellar mass are both most strongly peaked.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__MGE Basis__

We build a `Basis` of 30 linear Gaussians as the source-galaxy light model.

The Gaussians share a common centre (kept spherical here for simplicity) and have `sigma` values spaced in
log10 increments from 0.01" up to a reasonable size cap. The `intensity` of each Gaussian is a linear parameter,
solved for by linear algebra at fit time — no non-linear search is required.

The MGE centre matches the simulated source position from `simulator.py`.

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


source_bulge = build_source_basis(centre=(0.0, 0.0))

"""
The Gaussians cannot be plotted yet because their `intensity` values have not been solved for — linear light
profiles only acquire an `intensity` once a `FitImaging` runs its linear algebra step. After the fit below, we
visualise the source MGE basis with its solved-for intensities.

We set up the plotting grid we will use post-fit.
"""
plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

We now compose the two galaxies that form the lens system:

 - `lens` (z=0.5): a linear `lmp.Sersic` `bulge` which acts as BOTH the lens light AND the stellar mass
   component, coupled by `mass_to_light_ratio`. A spherical `NFWSph` `dark` matter halo aligned with the bulge.
   An `ExternalShear`.
 - `source` (z=1.0): the MGE basis above.

All non-linear parameters are set to the simulator's true values, so the fit visibly recovers the Einstein ring
without a search.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
        mass_to_light_ratio=0.2,
    ),
    dark=al.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.1, scale_radius=20.0),
    shear=al.mp.ExternalShear(gamma_1=-0.02, gamma_2=0.005),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=source_bulge,
)

"""
__Tracer__

The `Tracer` performs the ray-tracing. Internally it queries every mass profile attached to every galaxy in the
lens plane and sums their deflections. For our lens galaxy, this means the `bulge` contributes a stellar mass
deflection (its `lmp.Sersic` deflection scaled by `mass_to_light_ratio`), the `dark` halo contributes the
spherical NFW deflection, and `shear` contributes the external shear deflection — all summed before mapping
image-plane coordinates onto the source-plane.
"""
tracer = al.Tracer(galaxies=[lens, source])

"""
__Fit__

We pass the `Tracer` to a `FitImaging` to fit the dataset. The fit performs the ray-tracing (using the summed
stellar + dark + shear deflection), evaluates the source MGE at the source-plane, projects back to the image
plane, convolves with the PSF, and computes the residuals against the data.

The `linear_light_profile_intensity_dict` of the fit will hold a solved-for `intensity` for every Gaussian in
the source MGE basis.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Decomposed Deflection__

This is the section that makes the decomposed-mass fit conceptually distinct. The lens galaxy's total deflection
map is the SUM of three independent contributions:

  alpha_lens(theta) = alpha_stellar(theta)  +  alpha_dark(theta)  +  alpha_shear(theta)

where `alpha_stellar` is the `lmp.Sersic` bulge deflection (scaled internally by `mass_to_light_ratio`),
`alpha_dark` is the spherical NFW deflection, and `alpha_shear` is the external shear. Every individual
deflection is a public method on the corresponding profile.

We verify this by computing each contribution explicitly and confirming the sum equals what the full lens
galaxy returns.
"""
grid = dataset.grid

deflections_stellar = lens.bulge.deflections_yx_2d_from(grid=grid)
deflections_dark = lens.dark.deflections_yx_2d_from(grid=grid)
deflections_shear = lens.shear.deflections_yx_2d_from(grid=grid)

deflections_total_summed = deflections_stellar + deflections_dark + deflections_shear
deflections_total_lens = lens.deflections_yx_2d_from(grid=grid)

print(f"Stellar deflection (first 3): {deflections_stellar[:3]}")
print(f"Dark    deflection (first 3): {deflections_dark[:3]}")
print(f"Shear   deflection (first 3): {deflections_shear[:3]}")
print(f"Summed  deflection (first 3): {deflections_total_summed[:3]}")
print(f"Lens    deflection (first 3): {deflections_total_lens[:3]}")

"""
The same component-wise decomposition shows up in the convergence (kappa) map. Convergence is what is plotted
on log-scale "mass maps" in the literature, and the decomposed-mass approach lets us inspect the stellar and
dark contributions separately.
"""
kappa_stellar = lens.bulge.convergence_2d_from(grid=plot_grid)
kappa_dark = lens.dark.convergence_2d_from(grid=plot_grid)

aplt.plot_array(array=kappa_stellar, title="Stellar convergence (M/L * light)")
aplt.plot_array(array=kappa_dark, title="Dark matter convergence (NFW)")

"""
The lensed source's Einstein ring location is set by where `alpha_total` traces image-plane coordinates onto
the source position. Because the stellar contribution dominates inside the bulge and the dark contribution
takes over outside, the radial profile of the total deflection differs from any pure isothermal or power-law
model — this is the physical reason decomposed-mass fits constrain the M/L ratio and dark-halo concentration.

__Intensities__

After the fit, every linear Gaussian in the source MGE basis has been assigned an `intensity` via linear
algebra. These are available via the fit's `linear_light_profile_intensity_dict`, keyed by light profile object.

We print the intensity of the first Gaussian in the basis to confirm the source has been reconstructed.
"""
print(
    f"\nFirst Gaussian intensity, source = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

"""
A `Tracer` where every linear light profile has been replaced with an ordinary light profile carrying its
solved-for `intensity` is also accessible from the fit, which is useful for visualising the MGE basis with its
actual reconstructed amplitude.
"""
tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

subplot_basis_image(basis=tracer_fitted.galaxies[1].bulge, grid=plot_grid)

"""
__Wrap Up__

This script demonstrated the decomposed-mass API and the deflection decomposition, without invoking a non-linear
search. The lens galaxy's `bulge` simultaneously acts as a light profile and a stellar mass profile (coupled by
`mass_to_light_ratio`), and the separately-parameterized `dark` NFW halo adds the second mass contribution.
External shear is included for completeness; it contributes a small additional deflection.

In a real modeling workflow:

 - `modeling.py` shows how to fit the same system using `Nautilus`, but "cheats" by initialising priors at the
   true values. It is therefore only useful as a tutorial.
 - `chaining.py` is the practical workflow — two chained searches that fit the lens light first (treating it as
   a pure light profile), then reintroduce the stellar-mass coupling by re-using the bulge result as a
   `lmp.Sersic`. This is the script you'll actually use to fit data.
 - `slam.py` is the most robust pipeline for production-quality decomposed-mass modeling, chaining through
   SOURCE LP, SOURCE PIX, LIGHT LP, and MASS_LIGHT_DARK pipelines and ending in a pixelized source
   reconstruction.

The key takeaway from this script is that decomposed-mass lenses are fit with the same `Tracer` + `FitImaging`
objects as any other lens; the only difference is that the lens galaxy carries multiple independent mass
components (here: stellar via `lmp.Sersic`, dark via `NFWSph`, plus external shear) whose deflections sum into
the total lens-plane deflection.
"""
