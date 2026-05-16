"""
Features: Group Mass Stellar Dark Fit
=====================================

A group-scale strong lens where each main lens galaxy carries a decomposed mass model — a stellar component
tied to its observed light via a mass-to-light ratio, plus a separately-parameterized dark matter halo. The
total deflection at every image-plane coordinate is the sum over all main lens galaxies of the per-galaxy
stellar + dark contributions, plus a single external shear.

This script illustrates the API for performing a fit to a group-scale decomposed-mass lens via the standard
`Tracer` and `FitImaging` objects, without invoking a non-linear search. It is intended to make the per-galaxy
deflection decomposition concrete in the group context before the reader moves on to `modeling.py`
(search-based) or `chaining.py` / `slam.py` (realistic, robust modeling).

The source galaxy is modelled with a Multi Gaussian Expansion (MGE), the same source parameterization used in
`chaining.py` and `slam.py`.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Over Sampling:** Adaptive over-sampling at the main lens galaxy centres.
- **MGE Basis:** Build a `Basis` of linear Gaussians for the source.
- **Main Lens Centres:** Load the centres of the two main lens galaxies from JSON.
- **Galaxies:** Compose the lens galaxies (via a `lens_dict` loop) plus the MGE source.
- **Tracer:** Build the two-plane `Tracer` that performs the ray-tracing.
- **Fit:** Create a `FitImaging` and inspect the fit.
- **Decomposed Deflection (Multi-Galaxy):** A short tour of how the total lens-plane deflection is the sum,
  over every main lens galaxy, of stellar + dark contributions, plus the single external shear.
- **Intensities:** The solved-for linear light profile `intensity` values for each MGE Gaussian.
- **Wrap Up:** Summary and next steps.

__Prerequisites__

This script focuses on the API specific to a group-scale decomposed-mass fit. For background on the underlying
single-galaxy decomposition and the group `lens_dict` API, you should read first:

 - `autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/fit.py` — the single-galaxy
   decomposition tour. The Decomposed Deflection section below generalises that walkthrough across multiple
   main lens galaxies.
 - `autolens_workspace/scripts/group/start_here.py` — the group-scale `lens_dict` API, including how
   `main_lens_centres.json` is loaded and used to drive a per-galaxy loop.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py` — the MGE `Basis` API for
   the source.

The galaxy redshifts (`lenses=0.5`, `source=1.0`) and per-galaxy mass parameters match those used by the
simulator.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

"""
__Dataset__

Load and plot the strong lens dataset `mass_stellar_dark` via .fits files.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/mass_stellar_dark/simulator.py",
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

Load the centres of the two main lens galaxies from JSON. The same file is used by every other script in this
directory (modeling.py, chaining.py, slam.py).
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

Define a 3.7" circular mask, which includes both main lens galaxies and the lensed source emission.
"""
mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Apply adaptive over sampling at each main lens galaxy centre, so the stellar mass-to-light coupling is
evaluated accurately at the peak of each bulge.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__MGE Basis__

We build a `Basis` of 30 linear Gaussians as the source-galaxy light model, centred on the simulator's source
position.
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

plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

We compose each main lens galaxy via a `lens_dict` loop over `main_lens_centres`. Each galaxy gets:

 - a `lmp.Sersic` bulge (acts as light AND stellar mass via `mass_to_light_ratio`),
 - a `NFWSph` dark matter halo aligned with its bulge,
 - and (for the first lens galaxy only) an `ExternalShear` representing the group-wide shear field.

All non-linear parameters are set to the simulator's true values, so the fit visibly recovers the lensing
configuration without a search.
"""
bulge_params = [
    dict(axis_ratio=0.9, angle=45.0, intensity=1.0, effective_radius=0.8, m_to_l=0.20),
    dict(axis_ratio=0.8, angle=120.0, intensity=0.8, effective_radius=0.7, m_to_l=0.25),
]

dark_params = [
    dict(kappa_s=0.10, scale_radius=20.0),
    dict(kappa_s=0.08, scale_radius=20.0),
]

lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    galaxy_kwargs = dict(
        redshift=0.5,
        bulge=al.lmp.Sersic(
            centre=(centre[0], centre[1]),
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=bulge_params[i]["axis_ratio"],
                angle=bulge_params[i]["angle"],
            ),
            intensity=bulge_params[i]["intensity"],
            effective_radius=bulge_params[i]["effective_radius"],
            sersic_index=4.0,
            mass_to_light_ratio=bulge_params[i]["m_to_l"],
        ),
        dark=al.mp.NFWSph(
            centre=(centre[0], centre[1]),
            kappa_s=dark_params[i]["kappa_s"],
            scale_radius=dark_params[i]["scale_radius"],
        ),
    )

    if i == 0:
        galaxy_kwargs["shear"] = al.mp.ExternalShear(gamma_1=-0.02, gamma_2=0.005)

    lens_dict[f"lens_{i}"] = al.Galaxy(**galaxy_kwargs)

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

"""
__Tracer__

The `Tracer` performs the ray-tracing. Internally it queries every mass profile attached to every galaxy in
the lens plane and sums their deflections. For our group lens, this means each `lens_i`'s `bulge` contributes
a stellar mass deflection (`(M/L)_i * alpha_light_i`), each `dark` halo contributes an `NFWSph` deflection,
and `lens_0`'s `shear` contributes the external shear — all summed before mapping image-plane coordinates
onto the source-plane.
"""
tracer = al.Tracer(galaxies=list(lens_dict.values()) + [source])

"""
__Fit__

We pass the `Tracer` to a `FitImaging` to fit the dataset.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Decomposed Deflection (Multi-Galaxy)__

This is the section that makes the group-scale decomposed-mass fit conceptually distinct. The lens-plane
deflection map is the SUM, over every main lens galaxy AND every mass component in that galaxy, of independent
deflection contributions:

  alpha_lens(theta) = sum_i [ alpha_stellar_i(theta)  +  alpha_dark_i(theta) ]  +  alpha_shear(theta)
                    = sum_i [ (M/L)_i * alpha_light_i  +  alpha_NFW_i ]  +  alpha_shear

Every individual deflection is a public method on the corresponding profile.

We verify this by computing each contribution explicitly and confirming the sum equals what the `Tracer`
returns.
"""
grid = dataset.grid

alpha_stellar_list = [
    lens.bulge.deflections_yx_2d_from(grid=grid) for lens in lens_dict.values()
]
alpha_dark_list = [
    lens.dark.deflections_yx_2d_from(grid=grid) for lens in lens_dict.values()
]
alpha_shear = lens_dict["lens_0"].shear.deflections_yx_2d_from(grid=grid)

print(f"alpha_stellar[lens_0] (first coord): {alpha_stellar_list[0][0]}")
print(f"alpha_dark   [lens_0] (first coord): {alpha_dark_list[0][0]}")
print(f"alpha_stellar[lens_1] (first coord): {alpha_stellar_list[1][0]}")
print(f"alpha_dark   [lens_1] (first coord): {alpha_dark_list[1][0]}")
print(f"alpha_shear           (first coord): {alpha_shear[0]}")

alpha_total_summed = sum(alpha_stellar_list) + sum(alpha_dark_list) + alpha_shear

"""
The tracer-produced source-plane grid is just `grid - alpha_total_internal`. Recovering the internal total
deflection from `traced_grid_2d_list_from` and comparing to our hand-summed `alpha_total` is the cleanest
end-to-end check that we have accounted for every per-galaxy contribution.
"""
traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"alpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
The same component-wise decomposition shows up in the convergence (kappa) map. We sum each component's
contribution across all main lens galaxies and plot the result, which highlights that the stellar component is
peaked at the two galaxy centres while the dark halos extend more diffusely.
"""
kappa_stellar_total = sum(
    lens.bulge.convergence_2d_from(grid=plot_grid) for lens in lens_dict.values()
)
kappa_dark_total = sum(
    lens.dark.convergence_2d_from(grid=plot_grid) for lens in lens_dict.values()
)

aplt.plot_array(array=kappa_stellar_total, title="Stellar convergence (sum over galaxies)")
aplt.plot_array(array=kappa_dark_total, title="Dark matter convergence (sum over galaxies)")

"""
__Intensities__

After the fit, every linear Gaussian in the source MGE basis has been assigned an `intensity` via linear
algebra.
"""
print(
    f"\nFirst Gaussian intensity, source = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

subplot_basis_image(basis=tracer_fitted.galaxies[-1].bulge, grid=plot_grid)

"""
__Wrap Up__

This script demonstrated the group-scale decomposed-mass API and the per-galaxy deflection decomposition,
without invoking a non-linear search. Each main lens galaxy's `bulge` simultaneously acts as a light profile
and a stellar mass profile (coupled by its own `mass_to_light_ratio`), and each separately-parameterized
`dark` NFW halo adds an independent dark mass contribution. A single `ExternalShear` is attached to `lens_0`
representing the group-wide shear field.

In a real modeling workflow:

 - `modeling.py` shows how to fit the same system using `Nautilus`, but "cheats" by initialising priors at
   the true values. It is therefore only useful as a tutorial.
 - `chaining.py` is the practical workflow — two chained searches that fit the lens light first (treating each
   bulge as a pure light profile), then reintroduce the stellar-mass coupling and add the dark NFW halos.
 - `slam.py` is the most robust pipeline for production-quality group-scale decomposed-mass modeling, chaining
   through SOURCE LP, SOURCE PIX, LIGHT LP, and MASS_LIGHT_DARK pipelines and ending in a pixelized source
   reconstruction.

The key takeaway from this script is that a group-scale decomposed-mass lens is fit with the same `Tracer` +
`FitImaging` objects as any other lens; the only difference is that the lens plane carries MULTIPLE galaxies,
each with multiple independent mass components, all of whose deflections sum into the total lens-plane
deflection. The `lens_dict` API scales naturally to any number of main lens galaxies.
"""
