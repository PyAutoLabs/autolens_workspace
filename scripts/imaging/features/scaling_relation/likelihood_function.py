"""
__Log Likelihood Function: Scaling Relation__

Describes the one step of the `log_likelihood` computation that a scaling relation changes: how the lens plane's
deflection field is composed when most of the foreground galaxies' Einstein radii are not free parameters but
derived from the main lens's.

This script does NOT repeat the steps shared with single-plane lensing (mask, image-plane grid, PSF convolution,
chi-squared, noise normalisation, the linear-algebra solver for MGE source intensities). Those are documented in
the prerequisites and are entirely unaffected by the relation.

__Prerequisites__

 - `autolens_workspace/scripts/imaging/likelihood_function.py` — the canonical single-plane log-likelihood
   walkthrough.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/likelihood_function.py` — how a `Basis` of
   linear Gaussians is solved for via linear algebra.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the search-based version of the
   composition demonstrated here.

Sections covered in those scripts (e.g. "Chi Squared", "Noise Normalization Term", "Mapping Matrix") are not
repeated.

__What Changes For A Scaling Relation__

For a lens with a single mass profile, the lens-plane deflection is one profile evaluated per image-plane
coordinate:

    alpha(theta) = alpha_anchor(theta ; Isothermal parameters)

With a population of individually-modelled companions it becomes a sum:

    alpha(theta) = alpha_anchor(theta) + sum_i alpha_bounded_i(theta)

With the scaling tier active the sum extends again, but each new term's `einstein_radius` is a *function of a
parameter the model already has* rather than a parameter of its own:

    alpha(theta) = alpha_anchor(theta)
                 + sum_i alpha_bounded_i(theta)
                 + sum_j alpha_scaling_j(theta)

    where einstein_radius_j = einstein_radius_anchor * (L_j / L_anchor) ** 0.5

The consequence for the likelihood is that the number of deflection evaluations grows with the number of galaxies
while the dimensionality of the space being sampled does not. The tier costs likelihood *time*, not parameters —
worth knowing, because it is the one real cost of adding hundreds of galaxies this way.

Every other step is unchanged.

__Contents__

- **Dataset & Mask:** Standard set up (auto-simulating the dataset if absent).
- **Centres + Luminosities:** The three centre JSONs and the measured luminosities.
- **The Relation:** The per-galaxy Einstein radius, evaluated once.
- **Galaxies:** Mass-only lens-plane galaxies plus an MGE source.
- **Per-Galaxy Deflection:** Each tier's contribution to the deflection field.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Source-Plane Image:** The source MGE evaluated at the ray-traced grid.
- **Likelihood:** `FitImaging.log_likelihood`.
- **CSV Interface:** The same inputs read from a CSV instead.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset & Mask__
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/scaling_relation/simulator.py"],
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

Centres from one JSON per tier; luminosities as explicit Python lists (the CSV equivalent is at the end). In a real
analysis the luminosities are measured by a prior light-only fit — see `slam.py` in this folder.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
bounded_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

luminosity_anchor = 31.0962

bounded_galaxies_luminosities = [3.2595, 2.6076]

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

"""
__The Relation__

Evaluated here with the anchor's Einstein radius fixed at the simulator truth. Inside a model-fit the same
expression multiplies the free parameter instead, so the tier's radii move whenever the anchor's does — that
coupling is the whole mechanism.
"""
einstein_radius_anchor = 1.6
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the main lens.
    """
    return einstein_radius_anchor * (luminosity / luminosity_anchor) ** scaling_exponent


"""
__Galaxies__

Only mass matters for the deflection composition this script is about, so the lens-plane galaxies are mass-only and
the source carries a small MGE basis:

 - the **anchor** (z=0.5): `IsothermalSph` at the origin, `einstein_radius = 1.6`.
 - the **bounded tier** (z=0.5): two `IsothermalSph` companions.
 - the **scaling tier** (z=0.5): five `IsothermalSph` companions, radii from the relation.
 - the **source** (z=1.0): a basis of 10 linear Gaussians.

All profiles are **untruncated**: truncation encodes tidal stripping by a host halo, which a galaxy-scale lens does
not have. The truncated `dPIEMass` form of this tier belongs to the group and cluster workflows.
"""
total_gaussians = 10
log10_sigma_list = np.linspace(-2, np.log10(0.5), total_gaussians)

source_bulge = al.lp_basis.Basis(
    profile_list=[
        al.lp_linear.Gaussian(
            centre=(0.0, 0.1),
            ell_comps=(0.0, 0.0),
            sigma=10 ** log10_sigma_list[i],
        )
        for i in range(total_gaussians)
    ]
)

anchor = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius_anchor),
)

bounded_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, luminosity in zip(
        bounded_galaxies_centres, bounded_galaxies_luminosities
    )
]

scaling_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, luminosity in zip(
        scaling_galaxies_centres, scaling_galaxies_luminosities
    )
]

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

tracer = al.Tracer(
    galaxies=[anchor] + bounded_galaxies + scaling_galaxies + [source]
)

"""
__Per-Galaxy Deflection__

`Tracer.traced_grid_2d_list_from(...)` performs the image-plane -> source-plane ray-tracing, internally querying
every lens-plane mass profile and summing their deflections. Recomputing the same sum by hand makes the tier
structure explicit.
"""
masked_grid = dataset.grid

alpha_anchor = anchor.mass.deflections_yx_2d_from(grid=masked_grid)
alpha_bounded = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in bounded_galaxies
]
alpha_scaling = [
    g.mass.deflections_yx_2d_from(grid=masked_grid) for g in scaling_galaxies
]

alpha_total = alpha_anchor + sum(alpha_bounded) + sum(alpha_scaling)

print(f"alpha_anchor             (first coord): {alpha_anchor[0]}")
print(f"alpha_bounded (tier sum) (first coord): {sum(alpha_bounded)[0]}")
print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")
print(f"alpha_total   (all tiers) (first coord): {alpha_total[0]}")

"""
Each scaling-tier radius, and the relation that produced it:
"""
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: einstein_radius = "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Manual Ray-Tracing__

The source-plane grid is the image-plane grid minus the total deflection. Computed by hand and compared to the
`Tracer`'s, they agree to floating-point precision — the relation changes the *values* fed into the sum, not the
sum itself.
"""
grid_source_manual = masked_grid - alpha_total

traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_grid)
grid_source_tracer = traced_grid_list[1]

print(f"\nsource-plane grid (first coord, manual): {grid_source_manual[0]}")
print(f"source-plane grid (first coord, tracer): {grid_source_tracer[0]}")

assert np.allclose(np.asarray(grid_source_manual), np.asarray(grid_source_tracer))

"""
__Source-Plane Image__

The source MGE basis is evaluated at the ray-traced source-plane grid, each Gaussian carrying a placeholder
`intensity` that the linear-algebra step solves for.
"""
model_image_unconvolved = tracer.image_2d_from(grid=masked_grid)

aplt.plot_array(array=model_image_unconvolved, title="Model image before PSF convolution")

"""
So what `image_2d_from` does internally, for this two-tier population:

  1. Computes `alpha = alpha_anchor + sum_i alpha_bounded_i + sum_j alpha_scaling_j`, where each
     `alpha_scaling_j` comes from a profile whose `einstein_radius` was derived from
     `einstein_radius_anchor * (L_j / L_anchor) ** 0.5`.
  2. Ray-traces the image-plane grid: `grid_source = grid - alpha`.
  3. Evaluates the source MGE at `grid_source`.

Step 1 has `1 + len(bounded_galaxies) + len(scaling_galaxies)` terms — eight here — while the model being sampled
gained only three parameters per bounded galaxy and nothing at all for the scaling tier.

__Likelihood__

PSF convolution, the chi-squared and the noise normalisation are unchanged from the single-plane case, and the MGE
source's per-Gaussian intensities are solved by the linear-algebra step documented in the prerequisites. We hand
the remaining steps to `FitImaging`.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the scaling-relation fit: {fit.log_likelihood}")

"""
Do not read that number as a goodness-of-fit. The lens-plane galaxies here are deliberately mass-only, so none of
the foreground light present in the data is modelled and the residuals are dominated by it. `fit.py` includes the
foreground light and reports a sensible likelihood; this script exists to expose the deflection composition.

__CSV Interface__

The explicit luminosity lists above are the simplest interface. For larger populations,
`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts`:

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

Nothing downstream changes — the likelihood never sees where the numbers came from.
"""
scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")

print(f"Tier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The scaling-relation likelihood differs from an individually-modelled population in exactly one place: some
galaxies' `einstein_radius` values are set by the anchor's value and a measured luminosity rather than sampled.
Ray-tracing, source-plane evaluation, PSF convolution, chi-squared, noise normalisation and the linear algebra are
all shared with the standard imaging likelihood.

That is what keeps the model tractable as foreground galaxy count grows: 100 tied galaxies cost the same zero
parameters as five do, at the price of 100 extra deflection evaluations per likelihood call.
"""
