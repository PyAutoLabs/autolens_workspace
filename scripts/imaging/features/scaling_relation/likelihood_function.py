"""
__Log Likelihood Function: Scaling Relation__

This script describes the additional steps required to compute the `log_likelihood` for a strong lens whose
foreground galaxy population is split between two tiers — individually-modelled extras (each with its own free
`einstein_radius`) and scaling-tier extras (whose Einstein radii are derived from a shared reference-anchored
relation `einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`).

This script does NOT repeat the steps shared with single-plane lensing (mask, image-plane grid, PSF convolution,
chi-squared, noise normalization, linear-algebra solver for MGE source intensities). It documents only the part
of the likelihood function which is specific to a scaling-relation tier: the lens-plane deflection composition.

__Prerequisites__

The likelihood function below builds directly on the standard imaging and MGE likelihood functions. You should
read these notebooks first:

 - `autolens_workspace/scripts/imaging/likelihood_function.py` — the canonical single-plane log-likelihood
   walkthrough.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a `Basis`
   of linear Gaussians is solved for via linear algebra.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the search-based version of the
   composition demonstrated here.

Sections covered in those scripts (e.g. "Chi Squared", "Noise Normalization Term", "Mapping Matrix") are not
repeated here.

__Contents__

- **Prerequisites:** Reading order before this script (see above).
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Centres + Luminosities:** Load extras and scaling-tier galaxy data from JSON + CSV.
- **Galaxies:** Lens + individually-modelled extras + scaling-tier extras + MGE source.
- **Per-Galaxy Deflection:** Per-tier per-galaxy contributions, plus the scaling-relation evaluation.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Source-Plane Image:** Source MGE evaluated at the ray-traced grid.
- **Fit Check:** `FitImaging.log_likelihood`.
- **Wrap Up.**

__What Changes For A Scaling Relation__

For a single-component lens with one mass profile, the lens-plane deflection is just one profile evaluated at
each image-plane coordinate:

  alpha_lens(theta) = alpha_total(theta ; Isothermal parameters)

For a lens with a population of individually-modelled extras, the lens-plane deflection becomes a sum:

  alpha_lens(theta) = alpha_main(theta) + sum_i alpha_extra_individual_i(theta)

For a lens with BOTH tiers active (this script), the deflection sum extends to the scaling-tier extras, but each
scaling-tier galaxy's Einstein radius is NOT a free parameter — it is derived from a shared reference-anchored
relation and the galaxy's own luminosity:

  alpha_lens(theta) = alpha_main(theta)
                    + sum_i alpha_extra_individual_i(theta)
                    + sum_j alpha_extra_scaling_j(theta)

  where alpha_extra_scaling_j is the deflection of a mass profile whose
    einstein_radius_j = einstein_radius_ref * (luminosity_j / reference_luminosity) ** 0.5.

The model gains exactly 1 free parameter (`einstein_radius_ref`) regardless of how many galaxies
sit on the scaling-tier. Every other step of the likelihood (PSF convolution, chi-squared, noise normalization,
MGE linear-algebra solver) is unchanged.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the extra_and_scaling_galaxies dataset.
"""
dataset_name = "extra_and_scaling_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/imaging/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 6.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Centres + Luminosities__

Load the individually-modelled extras' centres from JSON and the scaling-tier extras' centres + luminosities
from CSV.
"""
individual_extras_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_extras_centres = scaling_table.centres
scaling_extras_luminosities = scaling_table.luminosities

"""
__Galaxies__

The three populations that participate in the ray-tracing:

 - `lens` (z=0.5): `IsothermalSph` mass at the origin with `einstein_radius=1.6` (simulator truth).
 - `individual_extras` (z=0.5): two `IsothermalSph` masses with simulator-true Einstein radii 0.4 and 0.5.
 - `scaling_extras` (z=0.5): two `IsothermalSph` masses with Einstein radii derived from the scaling relation
   `einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5` (simulator truth).
 - `source` (z=1.0): an MGE light component (a simple basis of 10 linear Gaussians).
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
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
)

individual_extras_einstein_radii = [0.4, 0.5]
individual_extras = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=tuple(centre), einstein_radius=er),
    )
    for centre, er in zip(individual_extras_centres, individual_extras_einstein_radii)
]

# reference_luminosity is an explicit fixed constant (Lenstool's "mag0"), not the
# sample max; einstein_radius_ref is the Einstein radius at that reference. Members
# share luminosity 0.45, so einstein_radius_ref * (0.45)**0.5 = 0.135 (simulator truth).
einstein_radius_ref = 0.2012
scaling_exponent = 0.5
reference_luminosity = 1.0

scaling_extras = []
for centre, luminosity in zip(scaling_extras_centres, scaling_extras_luminosities):
    einstein_radius = (
        einstein_radius_ref * (luminosity / reference_luminosity) ** scaling_exponent
    )
    scaling_extras.append(
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.IsothermalSph(
                centre=tuple(centre), einstein_radius=einstein_radius
            ),
        )
    )

source = al.Galaxy(redshift=1.0, bulge=build_source_basis(centre=(0.0, 0.1)))

tracer = al.Tracer(galaxies=[lens] + individual_extras + scaling_extras + [source])

"""
__Per-Galaxy Deflection__

The `Tracer.traced_grid_2d_list_from(...)` call performs the standard image-plane → source-plane ray-tracing.
Internally it queries every mass profile in the lens plane and sums their deflections.

To make the decomposition concrete we re-compute the same source-plane grid by hand. Each profile exposes its
own `deflections_yx_2d_from`; the SUM of the main lens deflection plus every per-galaxy contribution from BOTH
tiers is what the tracer applies internally:
"""
masked_grid = dataset.grid

alpha_lens = lens.mass.deflections_yx_2d_from(grid=masked_grid)
alpha_individual = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in individual_extras
]
alpha_scaling = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in scaling_extras
]

alpha_total = alpha_lens + sum(alpha_individual) + sum(alpha_scaling)

print(f"alpha_lens                       (first coord): {alpha_lens[0]}")
print(f"alpha_individual (tier sum)      (first coord): " f"{sum(alpha_individual)[0]}")
print(f"alpha_scaling    (tier sum)      (first coord): " f"{sum(alpha_scaling)[0]}")
print(f"alpha_total      (across all)    (first coord): {alpha_total[0]}")

"""
The scaling-tier contributions are computed from the scaling relation:
"""
for centre, luminosity in zip(scaling_extras_centres, scaling_extras_luminosities):
    er = einstein_radius_ref * (luminosity / reference_luminosity) ** scaling_exponent
    print(
        f"  scaling galaxy @ {tuple(centre)}: "
        f"einstein_radius = {einstein_radius_ref:.3f} * ({luminosity:.3f} / {reference_luminosity:.3f}) ** {scaling_exponent:.1f} = {er:.4f}"
    )

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
"""
model_image_unconvolved = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(
    array=model_image_unconvolved, title="Model image before PSF convolution"
)

"""
What `image_2d_from` does internally for our two-tier extras population:

  1. Computes `alpha_lens(theta) = alpha_main + sum_i alpha_extra_individual_i + sum_j alpha_extra_scaling_j`,
     where each `alpha_extra_scaling_j` is the deflection of a profile whose `einstein_radius` was derived from
     `einstein_radius_ref * (luminosity_j / reference_luminosity) ** 0.5`.
  2. Ray-traces the image-plane grid to obtain `grid_source = grid - alpha_lens`.
  3. Evaluates the source MGE at `grid_source`, producing its image-plane contribution.

For a single-lens system there is just one mass-profile contributing to step 1; for our mixed-strategy lens
there are `1 + len(individual_extras) + len(scaling_extras)` contributions, but the model only gains
`len(individual_extras)` free `einstein_radius` parameters plus 1 shared scaling parameter.

__Likelihood__

PSF convolution and chi-squared / noise normalization are unchanged from the single-plane case. The model image
above is convolved with the PSF and compared to the data via the standard imaging chi-squared expression
documented in `autolens_workspace/scripts/imaging/likelihood_function.py`. The MGE source's per-Gaussian
intensities are solved for via the linear-algebra step documented in
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py`.

We delegate the remaining steps to `FitImaging`, which assembles the full `log_likelihood`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the manual scaling-relation fit: {fit.log_likelihood}")

"""
__Wrap Up__

The scaling-relation `log_likelihood` differs from a population-of-individually-modelled-extras case in exactly
one place: some galaxies' `einstein_radius` values are not free parameters — they're derived from a shared
reference-anchored relation plus a per-galaxy luminosity. Every other step (ray-tracing, source-plane evaluation, PSF
convolution, chi-squared, noise normalization, linear algebra) is shared with the standard imaging likelihood
and documented in the prerequisite scripts.

This is what lets the model dimensionality stay tractable as foreground galaxy count grows: 100 scaling-tier
galaxies cost the same 1 shared parameter as 2 do.
"""
