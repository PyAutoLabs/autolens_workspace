"""
Features: Scaling Relation Fit
==============================

Fits the `scaling_relation` dataset with every parameter set to the simulator's truth, so the two-tier foreground
population can be inspected without a non-linear search in the way.

The point of this script is the **deflection sum**. A scaling relation does not change how ray-tracing works: the
lens plane's total deflection is still the sum of every mass profile's contribution. What the relation changes is
where the scaling tier's `einstein_radius` values *come from* — each is the main lens's Einstein radius scaled by a
luminosity ratio, rather than a number the search fits:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

Below, the relation is evaluated explicitly, each galaxy's deflection field is computed on its own, and the sum is
checked against what the `Tracer` produces internally.

__Prerequisites__

This script documents only what is specific to the scaling tier. Read these first:

 - `autolens_workspace/scripts/imaging/fit.py` — the standard single-plane fit.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the search-based version of the
   same composition, where the anchor's Einstein radius is a free parameter.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py` — the MGE source `Basis` API.

__Untruncated Profiles__

Every mass profile here is an **untruncated** `IsothermalSph`; truncation encodes tidal stripping by a host halo,
which a galaxy-scale lens does not have. The truncated `dPIEMass` form belongs to the group- and cluster-scale
workflows.

__Contents__

- **Dataset & Mask:** Load and mask the dataset (auto-simulating if absent).
- **Centres + Luminosities:** The three centre JSONs and the measured luminosities.
- **Over Sampling:** Adaptive over-sampling at every galaxy centre.
- **The Relation:** One function, evaluated per galaxy.
- **Galaxies:** Anchor, bounded tier, scaling tier, MGE source — all at simulator truth.
- **Tracer & Fit:** Build the `Tracer` and fit the dataset.
- **The Relation, Evaluated:** Each tied Einstein radius, shown as the arithmetic that produced it.
- **Deflection Sum:** Per-galaxy deflections, summed by hand and checked against the tracer.
- **Intensities:** The linear light profile `intensity` values solved for at the fit.
- **CSV Interface:** The same inputs read from a CSV instead.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
from autogalaxy.profiles.plot.basis_plots import subplot_image as subplot_basis_image

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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres + Luminosities__

One centre JSON per tier, and the measured luminosities as explicit Python lists — the interface worth reading
first. The CSV equivalent is at the end of this script.

In a real analysis these luminosities are measured by a prior light-only fit; `slam.py` in this folder is the
production path. Here they are the simulator's truth values.
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
__Over Sampling__
"""
all_galaxy_centres = (
    [tuple(c) for c in main_lens_centres]
    + [tuple(c) for c in bounded_galaxies_centres]
    + [tuple(c) for c in scaling_galaxies_centres]
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_galaxy_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__The Relation__

The anchor's Einstein radius is a fixed number here (the simulator truth) rather than a free parameter, so the
relation evaluates to a plain float per galaxy. In `modeling.py` the identical expression instead multiplies the
model's free `einstein_radius`, producing a tied parameter — same algebra, different object.
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

Four populations, all at the simulator's truth values:

 - the **anchor** (z=0.5): `SersicSph` light + `IsothermalSph` mass at the origin, `einstein_radius = 1.6`.
 - the **bounded tier** (z=0.5): two close companions. `modeling.py` frees their Einstein radii within a
   luminosity bound; the simulator placed them on the relation, so they are evaluated from it here too.
 - the **scaling tier** (z=0.5): five fainter companions, Einstein radii from the relation.
 - the **source** (z=1.0): an MGE basis of linear Gaussians.

The light intensities are the simulator's, so the residuals below should be pure noise.
"""
main_lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=1.5, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=einstein_radius_anchor),
)

bounded_galaxies_intensities = [0.5, 0.4]

bounded_galaxies = [
    al.Galaxy(
        redshift=0.5,
        bulge=al.lp.SersicSph(
            centre=tuple(centre),
            intensity=intensity,
            effective_radius=0.6,
            sersic_index=2.5,
        ),
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, intensity, luminosity in zip(
        bounded_galaxies_centres,
        bounded_galaxies_intensities,
        bounded_galaxies_luminosities,
    )
]

scaling_galaxies_intensities = [0.33, 0.24, 0.17, 0.11, 0.06]

scaling_galaxies = [
    al.Galaxy(
        redshift=0.5,
        bulge=al.lp.SersicSph(
            centre=tuple(centre),
            intensity=intensity,
            effective_radius=0.5,
            sersic_index=2.5,
        ),
        mass=al.mp.IsothermalSph(
            centre=tuple(centre), einstein_radius=einstein_radius_from(luminosity)
        ),
    )
    for centre, intensity, luminosity in zip(
        scaling_galaxies_centres,
        scaling_galaxies_intensities,
        scaling_galaxies_luminosities,
    )
]

total_gaussians = 30
log10_sigma_list = np.linspace(-4, np.log10(0.5), total_gaussians)

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

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

"""
__Tracer & Fit__

The `Tracer` queries every mass profile attached to every lens-plane galaxy and sums their deflections. Here that is
the anchor's profile plus one per bounded-tier galaxy plus one per scaling-tier galaxy — eight in total, from a
model that would cost only the anchor's own parameters plus three per bounded galaxy.
"""
tracer = al.Tracer(
    galaxies=[main_lens] + bounded_galaxies + scaling_galaxies + [source]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print(f"Log likelihood of the truth fit: {fit.log_likelihood}")

"""
__The Relation, Evaluated__

Each scaling-tier Einstein radius, shown as the arithmetic that produced it. Note how steeply luminosity falls
relative to Einstein radius — the relation goes as `L ** 0.5`, so the faintest member here is ~115x less luminous
than the anchor but still has ~9% of its Einstein radius.
"""
print(
    f"\nAnchor: einstein_radius = {einstein_radius_anchor:.4f}, L = {luminosity_anchor:.4f}"
)

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    einstein_radius = einstein_radius_from(luminosity)
    centre_str = f"({float(centre[0]):5.2f}, {float(centre[1]):5.2f})"
    print(
        f"  scaling galaxy @ {centre_str}: "
        f"{einstein_radius_anchor:.3f} * ({luminosity:.4f} / {luminosity_anchor:.4f}) ** {scaling_exponent} "
        f"= {einstein_radius:.4f}"
    )

"""
__Deflection Sum__

The lens-plane total deflection is the sum of every mass profile's contribution. We compute each one explicitly and
confirm the sum equals what the `Tracer` applies internally.
"""
grid = dataset.grid

alpha_anchor = main_lens.mass.deflections_yx_2d_from(grid=grid)
alpha_bounded = [g.mass.deflections_yx_2d_from(grid=grid) for g in bounded_galaxies]
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_galaxies]

print(f"\nalpha_anchor             (first coord): {alpha_anchor[0]}")
print(f"alpha_bounded (tier sum) (first coord): {sum(alpha_bounded)[0]}")
print(f"alpha_scaling (tier sum) (first coord): {sum(alpha_scaling)[0]}")

alpha_total_summed = alpha_anchor + sum(alpha_bounded) + sum(alpha_scaling)

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
It is worth reading those numbers carefully, because the naive intuition is wrong. An isothermal's deflection
*magnitude* is constant and equal to its Einstein radius, so a 0.35" member deflects by 0.35" everywhere — and the
tier's summed deflection above is a substantial fraction of the anchor's, not a rounding error.

What makes the tier a modest perturbation is not its deflection magnitude but the fact that a nearly *uniform*
deflection across the lensed images is degenerate with the source position: shift the source and you absorb it. The
physically meaningful quantity is the **differential** deflection each galaxy induces across the ring — a shear of
roughly `theta_E / 2d`, which for a 0.35" member 5" away is ~3%, and less for the fainter ones.

That is the honest case for this tier: individually each member is a percent-level shear, collectively they are a
real contribution to the deflection field, and the relation lets you include all of them for free.

__Intensities__

Every linear Gaussian in the source MGE basis has been assigned an `intensity` by linear algebra at the fit.
"""
print(
    f"\nFirst Gaussian intensity, source = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

subplot_basis_image(
    basis=tracer_fitted.galaxies[-1].bulge,
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05),
)

"""
__CSV Interface__

The explicit luminosity lists above are the simplest interface. For larger populations,
`al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`,
`.luminosities` and `.redshifts`, keeping centres and luminosities in a single file that cannot fall out of order:

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

The simulator writes one CSV per tier plus `main_lens_galaxies.csv` for the anchor, so the relation can be driven
entirely from CSVs.
"""
scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

print(f"\nTier luminosities from CSV: {list(scaling_table.luminosities)}")

"""
__Wrap Up__

Ray-tracing is unchanged by a scaling relation — the deflection field is still a plain sum over mass profiles, as
the assertion above confirms. The relation only determines what each scaling-tier `einstein_radius` is *set to*,
which is why 100 tied galaxies cost the model exactly as much as zero.

Next: `modeling.py` fits this composition with a search, and `slam.py` measures the luminosities it assumes.
"""
