"""
__Log Likelihood Function: Group Scaling Relation__

This script describes the additional steps required to compute the `log_likelihood` for a group-scale strong
lens whose foreground galaxy population is split across three tiers — main lens galaxies (modelled via the
group `lens_dict` API), individually-modelled extras (each with its own free `einstein_radius`), and
scaling-tier extras (whose Einstein radii are derived from a shared reference-anchored relation
`einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`, the Lenstool convention).

This script does NOT repeat the steps shared with single-plane lensing (mask, image-plane grid, PSF convolution,
chi-squared, noise normalization, linear-algebra solver for MGE source intensities). It documents only the part
of the likelihood function which is specific to the group three-tier API: the multi-tier lens-plane deflection
composition.

__Prerequisites__

 - `autolens_workspace/scripts/imaging/features/scaling_relation/likelihood_function.py` — the single-main-lens
   walkthrough. The three-tier version below generalises that example across the `lens_dict` API.
 - `autolens_workspace/scripts/group/start_here.py` — the group `lens_dict` API, including how
   `main_lens_centres.json` is loaded.
 - `autolens_workspace/scripts/imaging/likelihood_function.py` — canonical single-plane walkthrough.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a `Basis`
   of linear Gaussians is solved for via linear algebra.
 - `autolens_workspace/scripts/group/features/scaling_relation/modeling.py` — search-based three-tier version.

__Contents__

- **Prerequisites:** Reading order (see above).
- **Dataset & Mask.**
- **Centres + Luminosities:** Load each tier's data.
- **Galaxies:** lens_dict + extras + scaling-tier + MGE source.
- **Three-Tier Deflection:** Per-tier deflection sums plus the scaling-relation evaluation.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Source-Plane Image:** Source MGE evaluated at the ray-traced grid.
- **Fit Check.**
- **Wrap Up.**

__What Changes For A Group Three-Tier Scaling Relation__

For a single-galaxy lens with one mass profile, the lens-plane deflection is just one profile evaluated at
each image-plane coordinate:

  alpha_lens(theta) = alpha_total(theta ; Isothermal parameters)

For a group-scale lens with the three-tier API, the lens-plane deflection is the SUM across every main lens
galaxy AND every extras / scaling-tier galaxy:

  alpha_lens(theta) = sum_i alpha_main_lens_i(theta)
                    + sum_j alpha_extra_individual_j(theta)
                    + sum_k alpha_extra_scaling_k(theta)

  where alpha_extra_scaling_k is the deflection of a mass profile whose
    einstein_radius_k = einstein_radius_ref * (luminosity_k / reference_luminosity) ** 0.5.

The model gains exactly 1 free parameter (`einstein_radius_ref`) regardless of how many galaxies
sit on the scaling tier. Every other step of the likelihood (PSF convolution, chi-squared, noise normalization,
MGE linear-algebra solver) is unchanged.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the group scaling_relation dataset.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/scaling_relation/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 8.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Centres + Luminosities__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_galaxies_centres = scaling_table.centres
scaling_galaxies_luminosities = scaling_table.luminosities

"""
__Galaxies__

Three populations participate in the ray-tracing:

 - `lens_dict` (z=0.5): one `IsothermalSph` mass per main lens centre — here, just one with `einstein_radius=4.0`.
 - `individual_extras` (z=0.5): two `IsothermalSph` masses with simulator-true Einstein radii 0.8 and 1.0.
 - `scaling_extras` (z=0.5): two `IsothermalSph` masses with Einstein radii from the scaling relation.
 - `source` (z=1.0): a small MGE light component (10 linear Gaussians).
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


main_lens_einstein_radii = [4.0]

lens_dict = {}
for i, (centre, er) in enumerate(zip(main_lens_centres, main_lens_einstein_radii)):
    lens_dict[f"lens_{i}"] = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=tuple(centre), einstein_radius=er),
    )

individual_extras_einstein_radii = [0.8, 1.0]
individual_extras = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=tuple(centre), einstein_radius=er),
    )
    for centre, er in zip(extra_galaxies_centres, individual_extras_einstein_radii)
]

# reference_luminosity is an explicit fixed constant (Lenstool's reference
# magnitude "mag0"), not the sample max; einstein_radius_ref is the Einstein
# radius of a galaxy at that reference. Here L_ref = 1.0 (fiducial); both members
# share luminosity 0.45, so einstein_radius_ref * (0.45)**0.5 = 0.135 (simulator truth).
einstein_radius_ref = 0.2012
scaling_exponent = 0.5
reference_luminosity = 1.0

scaling_extras = []
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
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

tracer = al.Tracer(
    galaxies=list(lens_dict.values()) + individual_extras + scaling_extras + [source]
)

"""
__Three-Tier Deflection__

The `Tracer.traced_grid_2d_list_from(...)` call performs the standard image-plane → source-plane ray-tracing.
Internally it queries every mass profile across all three tiers and sums their deflections.

To make the decomposition concrete we re-compute the same source-plane grid by hand. Each profile exposes its
own `deflections_yx_2d_from`; the SUM of per-galaxy contributions across the three tiers is what the tracer
applies internally:
"""
masked_grid = dataset.grid

alpha_main_per_lens = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in lens_dict.values()
]
alpha_individual = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in individual_extras
]
alpha_scaling = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in scaling_extras
]

alpha_main_total = sum(alpha_main_per_lens)
alpha_individual_total = sum(alpha_individual)
alpha_scaling_total = sum(alpha_scaling)

alpha_total = alpha_main_total + alpha_individual_total + alpha_scaling_total

print(f"alpha_main_lens (tier sum, first coord)  : {alpha_main_total[0]}")
print(f"alpha_individual (tier sum, first coord) : {alpha_individual_total[0]}")
print(f"alpha_scaling    (tier sum, first coord) : {alpha_scaling_total[0]}")
print(f"alpha_total      (across all, first coord): {alpha_total[0]}")

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    er = einstein_radius_ref * (luminosity / reference_luminosity) ** scaling_exponent
    print(
        f"    scaling galaxy @ {tuple(centre)}: "
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
"""
model_image_unconvolved = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(
    array=model_image_unconvolved, title="Model image before PSF convolution"
)

"""
What `image_2d_from` does internally for our group-scale three-tier lens:

  1. Computes `alpha_lens(theta) = sum_i alpha_main_lens_i + sum_j alpha_extra_individual_j + sum_k alpha_extra_scaling_k`.
     Each `alpha_extra_scaling_k` is the deflection of a profile whose `einstein_radius` was derived from
     `einstein_radius_ref * (luminosity_k / reference_luminosity) ** 0.5`.
  2. Ray-traces the image-plane grid to obtain `grid_source = grid - alpha_lens`.
  3. Evaluates the source MGE at `grid_source`, producing its image-plane contribution.

For a single-galaxy lens there is just one profile contributing to step 1; for a group with M main lens
galaxies, N individually-modelled extras, and K scaling-tier extras, there are `M + N + K` contributions. The
model gains `M * (mass parameters per main lens) + N` free Einstein-radius parameters plus 1 shared scaling
normalization (`einstein_radius_ref`) — independent of K.

__Likelihood__

PSF convolution and chi-squared / noise normalization are unchanged from the single-plane case. The model image
above is convolved with the PSF and compared to the data via the standard imaging chi-squared expression
documented in `autolens_workspace/scripts/imaging/likelihood_function.py`. The MGE source's per-Gaussian
intensities are solved for via the linear-algebra step documented in
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py`.

We delegate the remaining steps to `FitImaging`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(
    f"\nLog likelihood of the manual group scaling-relation fit: {fit.log_likelihood}"
)

"""
__Wrap Up__

The group-scale scaling-relation `log_likelihood` differs from a single-main-lens scaling case in exactly one
place: the lens-plane deflection sums over multiple main lens galaxies in addition to the two extras tiers.
Every other step (ray-tracing, source-plane evaluation, PSF convolution, chi-squared, noise normalization,
linear algebra) is shared with the standard imaging likelihood.

This three-tier API is the production pattern for group-scale strong lenses with many foreground galaxies. It
lets the lens model stay tractable even when 10s or 100s of scaling-tier galaxies sit on the relation — the
lens-plane deflection is still a simple per-galaxy sum, but the scaling-tier contributions are parameterized
through luminosity rather than per-galaxy free parameters.
"""
