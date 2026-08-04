"""
__Log Likelihood Function: Multi Galaxy Scaling Relation__

Describes the one step of the `log_likelihood` computation that a brightest-galaxy-anchored scaling tier changes:
how the summed deflection field is composed when most of the deflectors' Einstein radii are not free parameters but
derived from the brightest co-dominant galaxy's.

This script does NOT repeat the steps shared with the multi-galaxy likelihood (mask, over-sampling, lens light, PSF
convolution, chi-squared, noise normalisation). Those are documented in the prerequisite and are unaffected by the
relation.

__Prerequisites__

 - `autolens_workspace/scripts/multi_galaxy/likelihood_function.py` — the multi-galaxy walkthrough, whose defining
   step is that the deflection field is the sum of every deflector's field. That step is exactly where this tier
   enters.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/likelihood_function.py` — the same material with a
   single-lens anchor.
 - `autolens_workspace/scripts/multi_galaxy/features/scaling_relation/modeling.py` — the search-based composition.

__What Changes For A Brightest-Galaxy-Anchored Tier__

The multi-galaxy likelihood already sums deflections over co-dominant deflectors:

    alpha(theta) = sum_d alpha_lens_d(theta)

Adding the scaling tier extends the sum, but every new term's `einstein_radius` is a function of a parameter the
model already has — the brightest galaxy's:

    alpha(theta) = sum_d alpha_lens_d(theta) + sum_j alpha_scaling_j(theta)

    where einstein_radius_j = einstein_radius_brightest * (L_j / L_brightest) ** 0.5
    and   brightest = argmax_d (L_d)

Two consequences specific to this regime:

 - The tier's radii are coupled to *one particular* deflector. If the brightest galaxy's Einstein radius moves during
   sampling, every member moves with it — so the tier is correlated with the brightest galaxy and not with the other
   co-dominant galaxy. That asymmetry is real and worth being aware of when reading a posterior.
 - The likelihood costs one deflection evaluation per member per call, so the tier costs *time*, not parameters.

Every other step is unchanged.

__Contents__

- **Dataset & Mask:** Standard set up (auto-simulating if absent).
- **Centres + Luminosities:** The pair, the tier, and identifying the brightest galaxy.
- **The Relation:** Anchored on the brightest galaxy.
- **Galaxies:** The co-dominant pair, the mass-only tier, and the source.
- **Summed Deflection Field:** Where the tier enters, computed term by term.
- **Manual Ray-Tracing:** Hand-compute the source-plane grid and confirm it matches `Tracer`.
- **Likelihood:** `FitImaging.log_likelihood`.
- **CSV Interface.**
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
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/scaling_relation/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Centres + Luminosities__

Explicit lists (the CSV equivalent is at the end). On real data these come from a light fit — see `slam.py`.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

main_lens_luminosities = [9.7913, 5.6663]

scaling_galaxies_luminosities = [1.2636, 0.8845, 0.6318, 0.3791, 0.2527]

brightest_index = int(np.argmax(main_lens_luminosities))
luminosity_brightest = main_lens_luminosities[brightest_index]

print(
    f"Brightest galaxy is lens_{brightest_index}, L_brightest = {luminosity_brightest}"
)

"""
__The Relation__
"""
einstein_radius_brightest = 1.0
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the brightest galaxy.
    """
    return (
        einstein_radius_brightest
        * (luminosity / luminosity_brightest) ** scaling_exponent
    )


"""
__Galaxies__

Mass-only throughout, since the deflection composition is what this script is about — the lens light steps are in the
prerequisite. The tier would be mass-only in a real fit here anyway, because its members sit outside the mask.

All profiles are **untruncated**: truncation encodes tidal stripping by a host halo, which this regime lacks.
"""
main_lens_mass_centres = [(0.30, 0.28), (-0.31, -0.22)]
main_lens_mass_angles = [45.0, 120.0]
main_lens_mass_axis_ratios = [0.85, 0.8]

main_lens_galaxies = [
    al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=main_lens_mass_centres[i],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=main_lens_mass_axis_ratios[i],
                angle=main_lens_mass_angles[i],
            ),
            einstein_radius=einstein_radius_from(main_lens_luminosities[i]),
        ),
    )
    for i in range(len(main_lens_luminosities))
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

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=main_lens_galaxies + scaling_galaxies + [source])

"""
__Summed Deflection Field__

The multi-galaxy step, extended by the tier. Each term is computed separately so the two populations are visible.
"""
masked_grid = dataset.grid

alpha_main = [
    galaxy.mass.deflections_yx_2d_from(grid=masked_grid)
    for galaxy in main_lens_galaxies
]
alpha_scaling = [
    galaxy.mass.deflections_yx_2d_from(grid=masked_grid) for galaxy in scaling_galaxies
]

alpha_total = sum(alpha_main) + sum(alpha_scaling)

for i, alpha in enumerate(alpha_main):
    print(f"alpha_lens_{i}             (first coord): {alpha[0]}")

print(f"alpha_scaling (tier sum)  (first coord): {sum(alpha_scaling)[0]}")
print(f"alpha_total   (all)       (first coord): {alpha_total[0]}")

"""
Each member's radius, and the relation that produced it. Every one of these is a function of
`einstein_radius_brightest`, so all five move together whenever the brightest galaxy's radius moves:
"""
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: einstein_radius = "
        f"{einstein_radius_brightest:.3f} * ({luminosity:.4f} / {luminosity_brightest:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Manual Ray-Tracing__

The source-plane grid is the image-plane grid minus the total deflection.
"""
grid_source_manual = masked_grid - alpha_total

traced_grid_list = tracer.traced_grid_2d_list_from(grid=masked_grid)
grid_source_tracer = traced_grid_list[1]

print(f"\nsource-plane grid (first coord, manual): {grid_source_manual[0]}")
print(f"source-plane grid (first coord, tracer): {grid_source_tracer[0]}")

assert np.allclose(np.asarray(grid_source_manual), np.asarray(grid_source_tracer))

"""
__Likelihood__

PSF convolution, chi-squared and noise normalisation are unchanged; `FitImaging` assembles the rest.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood: {fit.log_likelihood}")

"""
Do not read that number as a goodness-of-fit: the deflectors here are mass-only, so the pair's light is present in
the data but absent from the model and dominates the residuals. `fit.py` includes the light and reports a sensible
likelihood.

__CSV Interface__

`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts`:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    main_lens_luminosities = main_lens_table.luminosities

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_luminosities = scaling_table.luminosities

The `argmax` identifying the brightest galaxy runs on those luminosities unchanged.
"""
scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

print(f"Tier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The tier changes one thing in the likelihood: some deflectors' `einstein_radius` values are set by the brightest
galaxy's value and a measured luminosity rather than sampled. The summation, ray-tracing, convolution, chi-squared and
noise normalisation are all shared with the multi-galaxy likelihood.

The cost is one extra deflection evaluation per member per likelihood call — and zero extra parameters.
"""
