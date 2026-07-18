"""
__Log Likelihood Function: Group Mass Stellar Dark__

This script describes the additional steps required to compute the `log_likelihood` for a group-scale strong
lens whose mass model decomposes each main lens galaxy's mass into a stellar component (tied to its light via
a mass-to-light ratio) and a separately-parameterized dark matter halo.

This script does NOT repeat the steps shared with single-galaxy lensing (mask, image-plane grid, PSF
convolution, chi-squared, noise normalization, linear-algebra solver for MGE source intensities). It documents
only the parts of the likelihood function which are specific to a group-scale decomposed-mass lens: the
multi-galaxy deflection composition.

__Prerequisites__

The likelihood function below builds directly on the single-galaxy decomposed-mass likelihood and the standard
imaging / MGE likelihood functions. You should read these notebooks first:

 - `autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/likelihood_function.py` — the
   single-galaxy decomposed-mass walkthrough, covering the lens-plane deflection composition
   `(M/L) * alpha_light + alpha_NFW + alpha_shear` for one galaxy. This script generalises that walkthrough
   across multiple main lens galaxies.
 - `autolens_workspace/scripts/group/start_here.py` — the group-scale `lens_dict` API, including how
   `main_lens_centres.json` is loaded.
 - `autolens_workspace/scripts/imaging/likelihood_function.py` — the canonical single-plane log likelihood,
   covering image-plane grids, ray-tracing, source-plane evaluation, PSF convolution, chi-squared and the
   noise normalization term.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a
   `Basis` of linear Gaussians is solved for via linear algebra.

Sections covered in those scripts (e.g. "Chi Squared", "Noise Normalization Term", "Mapping Matrix") are not
repeated here.

__Contents__

- **Prerequisites:** Reading order before this script (see above).
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Main Lens Centres:** Load the centres of the two main lens galaxies from JSON.
- **Galaxies:** Build the `lens_dict` and the MGE source.
- **Decomposed Deflection (Multi-Galaxy):** Sum of per-galaxy stellar + dark + shear deflections.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Source-Plane Image:** Source MGE evaluated at the ray-traced grid.
- **Model Image:** Reference up to the canonical chi-squared / noise normalization.
- **Fit Check:** `FitImaging.log_likelihood`.
- **Wrap Up.**

__What Changes For A Group-Scale Decomposed Mass Model__

For a single-galaxy lens with a single total-mass profile (e.g. `Isothermal`), the lens-plane deflection at
every image-plane coordinate is produced by ONE mass profile:

  alpha_lens(theta) = alpha_total(theta ; Isothermal parameters)

For a single-galaxy DECOMPOSED-mass lens, the same galaxy carries multiple independent mass components and
the lens-plane deflection is their sum (see the single-galaxy prerequisite above):

  alpha_lens(theta) = (M/L) * alpha_light(theta)  +  alpha_NFW(theta)  +  alpha_shear(theta)

For a GROUP-SCALE decomposed-mass lens, every main lens galaxy carries its own decomposition, and the
lens-plane deflection is the SUM of every galaxy's contribution:

  alpha_lens(theta) = sum_i [ (M/L)_i * alpha_light_i(theta)  +  alpha_NFW_i(theta) ]  +  alpha_shear(theta)

A single `ExternalShear` is attached to `lens_0` representing the group-wide shear field. Every other step of
the likelihood (PSF convolution, chi-squared, noise normalization, MGE linear-algebra solver) is unchanged.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the group mass_stellar_dark dataset.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "group" / dataset_name

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

mask_radius = 3.7

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Main Lens Centres__

Load the two main lens galaxy centres from JSON, the same file used by every other script in this directory.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Galaxies__

The main lens galaxies + the source that participate in the ray-tracing:

 - Each `lens_i` (z=0.5): an `lmp.Sersic` bulge (acting as light + stellar mass via a single
   `mass_to_light_ratio`), an `NFWSph` dark matter halo aligned with the bulge. Only `lens_0` carries an
   `ExternalShear`.
 - `source` (z=1.0): an MGE light component (a simple basis of 10 linear Gaussians).

The mass-profile parameters are set to the simulator's true values so the manual likelihood computation below
produces a sensible-looking model image.
"""
total_gaussians = 10
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

source = al.Galaxy(redshift=1.0, bulge=build_source_basis(centre=(0.0, 0.0)))

tracer = al.Tracer(galaxies=list(lens_dict.values()) + [source])

"""
__Decomposed Deflection (Multi-Galaxy)__

The single `Tracer.traced_grid_2d_list_from(...)` call performs the standard image-plane → source-plane
ray-tracing. Internally it queries every mass profile on every galaxy in the lens plane and sums their
deflections.

To make the decomposition concrete we re-compute the same source-plane grid by hand. Each profile exposes its
own `deflections_yx_2d_from`; the SUM of all per-galaxy stellar + dark contributions, plus the single external
shear, is what the tracer applies internally:
"""
masked_grid = dataset.grid

alpha_stellar_list = [
    lens.bulge.deflections_yx_2d_from(grid=masked_grid) for lens in lens_dict.values()
]
alpha_dark_list = [
    lens.dark.deflections_yx_2d_from(grid=masked_grid) for lens in lens_dict.values()
]
alpha_shear = lens_dict["lens_0"].shear.deflections_yx_2d_from(grid=masked_grid)

alpha_total = sum(alpha_stellar_list) + sum(alpha_dark_list) + alpha_shear

print(f"alpha_stellar[lens_0] (first coord): {alpha_stellar_list[0][0]}")
print(f"alpha_dark   [lens_0] (first coord): {alpha_dark_list[0][0]}")
print(f"alpha_stellar[lens_1] (first coord): {alpha_stellar_list[1][0]}")
print(f"alpha_dark   [lens_1] (first coord): {alpha_dark_list[1][0]}")
print(f"alpha_shear           (first coord): {alpha_shear[0]}")
print(f"alpha_total           (first coord): {alpha_total[0]}")

"""
__Manual Ray-Tracing__

The source-plane grid is the image-plane grid minus the total deflection. We compute it by hand and compare to
the `Tracer`-produced grid; they should be identical to within floating-point precision.
"""
grid_source_manual = masked_grid - alpha_total

traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_grid)
grid_source_tracer = traced_grid_list[1]

print(f"\nsource-plane grid (first coord, manual): {grid_source_manual[0]}")
print(f"source-plane grid (first coord, tracer): {grid_source_tracer[0]}")

assert np.allclose(np.asarray(grid_source_manual), np.asarray(grid_source_tracer))

"""
__Source-Plane Image__

The source galaxy's MGE basis is evaluated at the ray-traced source-plane grid, producing image-plane pixel
values per Gaussian with a placeholder `intensity=1.0`. The true `intensity` of each Gaussian is solved for at
the linear-algebra step (see the MGE likelihood prerequisite).

For this manual walkthrough we use the convenience method `image_2d_from` on the `Tracer`, which evaluates the
source MGE at the correct (ray-traced) plane and projects it into the image plane.
"""
model_image_unconvolved = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(
    array=model_image_unconvolved, title="Model image before PSF convolution"
)

"""
What `image_2d_from` does internally for our group-scale decomposed-mass lens:

  1. Computes `alpha_lens(theta) = sum_i [ alpha_stellar_i + alpha_dark_i ] + alpha_shear` (the decomposition
     above).
  2. Ray-traces the image-plane grid to obtain `grid_source = grid - alpha_lens`.
  3. Evaluates the source MGE at `grid_source`, producing its image-plane contribution.

For a single-galaxy lens there is just one galaxy contributing to step 1; for a group with N main lens
galaxies there are 2N + 1 mass contributions (stellar + dark per galaxy, plus one shear).

__Model Image__

PSF convolution and chi-squared / noise normalization are unchanged from the single-plane case. The model
image above is convolved with the PSF and compared to the data via the standard imaging chi-squared expression
documented in `autolens_workspace/scripts/imaging/likelihood_function.py`. The MGE source's per-Gaussian
intensities are solved for via the linear-algebra step documented in
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py`.

We delegate the remaining steps to `FitImaging`, which handles the linear-algebra step that solves for each
Gaussian's `intensity` and assembles the full `log_likelihood`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(
    f"\nLog likelihood of the manual group mass-stellar-dark fit: {fit.log_likelihood}"
)

"""
__Likelihood__

The final `log_likelihood` combines:

  - The chi-squared term, computed from the residuals between the PSF-convolved model image and the data,
    weighted by the noise map.
  - The noise normalization term, the standard Gaussian normalization over all unmasked pixels.
  - The linear algebra terms (regularization and curvature determinants) introduced by the MGE
    `Basis` of linear Gaussians.

The first two are documented in `imaging/likelihood_function.py`; the third in
`imaging/features/multi_gaussian_expansion/likelihood_function.py`. No new terms are introduced by the
group-scale decomposed mass model — the only change is the lens-plane deflection composition described above.

__Wrap Up__

The group-scale decomposed-mass `log_likelihood` differs from the single-galaxy decomposed-mass case in exactly
one place: the lens-plane deflection is a sum over MULTIPLE galaxies, each contributing its own stellar +
dark decomposition, plus a single external shear. Every other step (ray-tracing, source-plane evaluation, PSF
convolution, chi-squared, noise normalization, linear algebra) is shared with the standard imaging likelihood
and documented in the prerequisite scripts.

This per-galaxy decomposition is the standard tool for studying mass-to-light variation across a group
environment: each `mass_to_light_ratio` is constrained independently by the data, so the relative stellar vs
dark contribution of each main lens galaxy can be measured separately.
"""
