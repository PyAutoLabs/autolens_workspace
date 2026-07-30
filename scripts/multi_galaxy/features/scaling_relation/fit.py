"""
Features (Multi Galaxy): Scaling Relation Fit
============================================

Fits the multi-galaxy `scaling_relation` dataset at the simulator's truth values, so the tied tier can be inspected
without a non-linear search in the way.

Two things are worth watching here that the galaxy-scale version of this script cannot show:

 1. **The anchor is chosen, not assumed.** The relation is anchored on the *brightest* co-dominant deflector, found
    by `argmax` over the measured luminosities.
 2. **The tier competes with a second co-dominant deflector.** The deflection sum below has two large contributions
    rather than one, and the tier's collective contribution can be compared against both.

__Prerequisites__

 - `autolens_workspace/scripts/multi_galaxy/fit.py` — the multi-galaxy fit this extends.
 - `autolens_workspace/scripts/multi_galaxy/features/scaling_relation/modeling.py` — the search-based version.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/fit.py` — the fuller walkthrough of the relation
   itself, with a single-lens anchor.

__Untruncated Profiles__

All mass profiles are **untruncated**: truncation encodes tidal stripping by a host halo's potential, which a
multi-galaxy lens does not have. Truncated `dPIEMass` members belong to the group and cluster workflows.

__Contents__

- **Dataset & Mask:** Load and mask (auto-simulating if absent).
- **Centres + Luminosities:** The pair, the tier, and identifying the brightest galaxy.
- **The Relation:** Anchored on the brightest galaxy.
- **Galaxies:** The co-dominant pair, the mass-only tier, and the source.
- **Tracer & Fit:** Build the `Tracer` and fit.
- **Deflection Sum:** Per-galaxy deflections, summed by hand and checked against the tracer.
- **How Much Does The Tier Matter?:** The tier's contribution against the pair's.
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres + Luminosities__

Explicit lists (the CSV equivalent is at the end). On real data these luminosities come from a light fit — see
`slam.py` in this folder.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

main_lens_luminosities = [9.7913, 5.6663]

scaling_galaxies_luminosities = [1.2636, 0.8845, 0.6318, 0.3791, 0.2527]

brightest_index = int(np.argmax(main_lens_luminosities))
luminosity_brightest = main_lens_luminosities[brightest_index]

print(f"Brightest galaxy is lens_{brightest_index}, L_brightest = {luminosity_brightest}")

"""
__The Relation__

Anchored on the brightest galaxy's Einstein radius, which is a fixed number here (simulator truth) and a free
parameter in `modeling.py`.
"""
einstein_radius_brightest = 1.0
scaling_exponent = 0.5


def einstein_radius_from(luminosity):
    """
    The Faber-Jackson Einstein radius of a galaxy of the input luminosity, anchored on the brightest galaxy.
    """
    return einstein_radius_brightest * (luminosity / luminosity_brightest) ** scaling_exponent


"""
__Galaxies__

The co-dominant pair at simulator truth, then the mass-only tier. The pair's Einstein radii both come from the
relation in the simulator — but note that `modeling.py` still *frees* them, because a co-dominant deflector should
not be constrained by a scaling law.

The tier is mass-only because its members sit 5.5-7" out, outside the 3.0" mask, so their light is not in the fit.
"""
main_lens_bulges = [
    al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
]

main_lens_mass_centres = [(0.30, 0.28), (-0.31, -0.22)]
main_lens_mass_angles = [45.0, 120.0]
main_lens_mass_axis_ratios = [0.85, 0.8]

main_lens_galaxies = [
    al.Galaxy(
        redshift=0.5,
        bulge=main_lens_bulges[i],
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

"""
__Tracer & Fit__
"""
tracer = al.Tracer(galaxies=main_lens_galaxies + scaling_galaxies + [source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"\nLog likelihood of the truth fit: {fit.log_likelihood}")

"""
Each member's Einstein radius, as the arithmetic that produced it:
"""
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: "
        f"{einstein_radius_brightest:.3f} * ({luminosity:.4f} / {luminosity_brightest:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius_from(luminosity):.4f}"
    )

"""
__Deflection Sum__

The lens-plane total deflection is the sum over every mass profile — here two co-dominant deflectors plus five tied
members.
"""
grid = dataset.grid

alpha_main = [g.mass.deflections_yx_2d_from(grid=grid) for g in main_lens_galaxies]
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_galaxies]

for i, alpha in enumerate(alpha_main):
    print(f"\nalpha_lens_{i}            (first coord): {alpha[0]}")

print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")

alpha_total_summed = sum(alpha_main) + sum(alpha_scaling)

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
__How Much Does The Tier Matter?__

The honest comparison, since this tier is explicitly *not* a standard ingredient at multi-galaxy scale. Below, the
tier's summed deflection magnitude is compared against the co-dominant pair's, averaged over the mask.

Read the result carefully. An isothermal's deflection magnitude is constant and equal to its Einstein radius, so
five members at 0.16-0.36" are not a negligible *absolute* deflection. But a nearly uniform deflection across the
lensed images is degenerate with the source position — shift the source and you absorb most of it. What survives is
the differential deflection across the ring, a shear of roughly `theta_E / 2d`, which is ~2.5% for the closest
member here. That is the number that decides whether this tier is worth the trouble.
"""
magnitude_main = np.mean(np.linalg.norm(np.asarray(sum(alpha_main)), axis=1))
magnitude_scaling = np.mean(np.linalg.norm(np.asarray(sum(alpha_scaling)), axis=1))

print(f"\nMean |alpha| from the co-dominant pair: {magnitude_main:.4f}")
print(f"Mean |alpha| from the scaling tier:    {magnitude_scaling:.4f}")
print(f"Ratio:                                 {magnitude_scaling / magnitude_main:.4f}")

"""
__CSV Interface__

`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts` — a single file per tier that cannot fall out of order:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    main_lens_luminosities = main_lens_table.luminosities

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

The `argmax` that identifies the brightest galaxy runs on those luminosities unchanged.
"""
main_lens_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "main_lens_galaxies.csv"
)

print(f"\nMain lens luminosities from CSV: {list(main_lens_table.luminosities)}")

"""
__Wrap Up__

Ray-tracing is unchanged by the relation — the deflection field is a plain sum over mass profiles, as the assertion
confirms. What the relation fixes is what each member's `einstein_radius` is *set to*, anchored on whichever
co-dominant deflector is brightest.

Next: `modeling.py` fits this with a search, and `slam.py` measures the luminosities it assumed.
"""
