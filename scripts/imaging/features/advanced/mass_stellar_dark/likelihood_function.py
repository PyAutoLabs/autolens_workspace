"""
__Log Likelihood Function: Mass Stellar Dark__

This script describes the additional steps required to compute the `log_likelihood` for a strong lens whose
mass model decomposes the lens galaxy's mass into a stellar component (tied to its light via a mass-to-light
ratio) and a separately-parameterized dark matter halo.

This script does NOT repeat the steps shared with single-plane lensing (mask, image-plane grid, PSF
convolution, chi-squared, noise normalization, linear-algebra solver for MGE source intensities). It documents
only the part of the likelihood function which is specific to a decomposed-mass lens: the lens-plane
deflection composition.

__Prerequisites__

The likelihood function below builds directly on the standard imaging and MGE likelihood functions. You should
read these notebooks first:

 - `autolens_workspace/scripts/imaging/likelihood_function.py` — the canonical single-plane log likelihood
   walkthrough, covering image-plane grids, ray-tracing, source-plane evaluation, PSF convolution, chi-squared
   and the noise normalization term.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a `Basis`
   of linear Gaussians is solved for via linear algebra.

Sections covered in those scripts (e.g. "Chi Squared", "Noise Normalization Term", "Mapping Matrix") are not
repeated here; this script focuses entirely on what changes for a decomposed mass model.

__Contents__

- **Prerequisites:** Reading order before this script (see above).
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Galaxies:** A decomposed-mass lens (stellar + dark + shear) and an MGE source.
- **Decomposed Deflection:** The lens-plane deflection sum that produces the ray-traced source-plane grid.
- **Source-Plane Image:** Source MGE evaluated at the ray-traced grid.
- **Model Image:** PSF convolution and reference up to the canonical chi-squared / noise normalization.
- **Fit Check:** Confirm the manual ray-tracing matches `Tracer.traced_grid_2d_list_from` and the model fit
  produces a finite `log_likelihood`.
- **Wrap Up.**

__What Changes For A Decomposed Mass Model__

For a single-plane lens with a total-mass profile such as `Isothermal` or `PowerLaw`, the lens-plane deflection
at every image-plane coordinate is produced by ONE mass profile:

  alpha_lens(theta) = alpha_total(theta ; Isothermal parameters)

For a decomposed mass model, the lens galaxy carries multiple independent mass components, and the lens-plane
deflection is their SUM:

  alpha_lens(theta) = alpha_stellar(theta) + alpha_dark(theta) + alpha_shear(theta)
                    = (M/L) * alpha_light(theta ; bulge parameters)
                       + alpha_NFW(theta ; kappa_s, scale_radius)
                       + alpha_shear(theta ; gamma_1, gamma_2)

The stellar contribution comes from the lens galaxy's light profile, scaled by the `mass_to_light_ratio`
parameter — i.e. the lens light is converted into a stellar surface density before being turned into a
deflection. The dark contribution is the NFW deflection of the dark matter halo. External shear contributes
a small uniform deflection set by the two shear components.

Every other step of the likelihood (PSF convolution, chi-squared, noise normalization, MGE linear-algebra
solver) is unchanged.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the mass_stellar_dark dataset. The auto-simulation block mirrors the other example scripts.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "imaging" / dataset_name

if not dataset_path.exists():
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

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxies__

The two galaxies that participate in the ray-tracing:

 - `lens` (z=0.5): a linear `lmp.Sersic` bulge (acting as light + stellar mass via a single `mass_to_light_ratio`),
   an `NFWSph` dark matter halo aligned with the bulge, and an `ExternalShear`.
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
    bulge=build_source_basis(centre=(0.0, 0.0)),
)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Decomposed Deflection__

The single call below performs the standard image-plane → source-plane ray-tracing.
`traced_grid_2d_list_from` returns one grid per plane: the image-plane grid (no deflection) and the
source-plane grid (deflected by every mass profile in the lens plane, summed).

To make the decomposition concrete we re-compute the same source-plane grid by hand. Each mass profile exposes
its own `deflections_yx_2d_from` method; the SUM of those three deflection maps is what the tracer applies
internally to produce the source-plane grid:
"""
masked_grid = dataset.grid

deflections_stellar = lens.bulge.deflections_yx_2d_from(grid=masked_grid)
deflections_dark = lens.dark.deflections_yx_2d_from(grid=masked_grid)
deflections_shear = lens.shear.deflections_yx_2d_from(grid=masked_grid)

deflections_total = deflections_stellar + deflections_dark + deflections_shear

grid_source_manual = masked_grid - deflections_total

print(f"deflections_stellar (first coord): {deflections_stellar[0]}")
print(f"deflections_dark    (first coord): {deflections_dark[0]}")
print(f"deflections_shear   (first coord): {deflections_shear[0]}")
print(f"deflections_total   (first coord): {deflections_total[0]}")
print(f"source-plane grid (first coord, manual): {grid_source_manual[0]}")

"""
We compare the hand-summed source-plane grid to the one produced by the `Tracer`, confirming the deflection
decomposition reproduces the internal ray-tracing exactly:
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_grid)
grid_source_tracer = traced_grid_list[1]

print(f"source-plane grid (first coord, tracer): {grid_source_tracer[0]}")

"""
__Source-Plane Image__

The source galaxy's MGE basis is evaluated at the ray-traced source-plane grid, producing image-plane (y,x)
pixel values per Gaussian with a placeholder `intensity=1.0`. The true `intensity` of each Gaussian is solved
for at the linear-algebra step (see the MGE likelihood prerequisite).

For this manual walkthrough we use the convenience method `image_2d_from` on the `Tracer`, which evaluates the
source MGE at the correct (ray-traced) plane and projects it into the image plane.
"""
model_image_unconvolved = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(array=model_image_unconvolved, title="Model image before PSF convolution")

"""
What `image_2d_from` does internally for our decomposed-mass lens:

  1. Computes `alpha_lens(theta) = alpha_stellar + alpha_dark + alpha_shear` (the decomposition above).
  2. Ray-traces the image-plane grid to obtain `grid_source = grid - alpha_lens`.
  3. Evaluates the source MGE at `grid_source`, producing its image-plane contribution.
  4. (If the lens has a light component, also evaluates it at the image-plane grid and adds to the model image.)

For a single-component total-mass lens there is just one profile contributing to step 1; for the decomposed
mass model there are three.

__Model Image__

PSF convolution and chi-squared / noise normalization are unchanged from the single-plane case. The model image
above is convolved with the PSF and compared to the data via the standard imaging chi-squared expression
documented in `autolens_workspace/scripts/imaging/likelihood_function.py`. The MGE source's per-Gaussian
intensities are solved for via the linear-algebra step documented in
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py`.

We delegate the remaining steps to `FitImaging`, which handles the linear-algebra step that solves for each
Gaussian's `intensity` and assembles the full `log_likelihood`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the manual mass-stellar-dark fit: {fit.log_likelihood}")

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
decomposed mass model — the only change is the lens-plane deflection composition described above.

__Wrap Up__

The decomposed-mass `log_likelihood` differs from a single-component total-mass case in exactly one place: the
lens-plane deflection is a sum of three independent contributions (stellar via `(M/L) * alpha_light`, dark via
`alpha_NFW`, and shear via `alpha_shear`) rather than a single `alpha_total`. Every other step (ray-tracing,
PSF convolution, chi-squared, noise normalization, linear algebra) is shared with the standard imaging
likelihood and documented in the prerequisite scripts.

The mass-to-light coupling between the stellar component and the lens light is what makes decomposed-mass fits
informative about the lens's stellar mass: the same `bulge` parameters that determine the observed light profile
also determine the stellar deflection, so the data constrains the `mass_to_light_ratio` directly. The dark NFW
contribution then fills in whatever deflection the stellar component cannot account for — which is what the
science measurement of "stellar vs dark matter contribution to the lens" rests on.
"""
