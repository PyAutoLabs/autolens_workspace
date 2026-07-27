"""
Features: Scaling Relation Fit
==============================

A strong lens system often has many foreground galaxies near the line of sight to the source, in addition to the
primary lens. As the number of foreground galaxies grows, modelling each one individually with its own free
`einstein_radius` parameter rapidly becomes intractable.

A common solution is to split foreground galaxies into two tiers:

 - **Individually-modelled extras** — the brighter, closer companions that contribute non-trivially to the lensing
   on their own. Each gets its own free Einstein radius.
 - **Scaling-relation extras** — the long tail of fainter companions whose Einstein radii are tied together via a
   shared reference-anchored relation
     einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5
   so adding more galaxies to this tier does not grow the model.

This script illustrates the API for performing a fit to a strong lens with both tiers active, via the standard
`Tracer` and `FitImaging` objects, without invoking a non-linear search. It is intended to make the per-galaxy
deflection composition concrete before the reader moves on to `modeling.py` (search-based).

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Over Sampling:** Adaptive over-sampling at every galaxy centre.
- **Centres + Luminosities:** Load extra-galaxy centres (JSON) and scaling-tier centres + luminosities (CSV).
- **MGE Basis:** Build a `Basis` of linear Gaussians for the source.
- **Galaxies:** Concrete composition — lens, individually-modelled extras, scaling-tier extras, source.
- **Tracer:** Build the `Tracer` and fit the dataset.
- **Scaling Relation Tour:** Per-galaxy deflections sum into the tracer's total deflection. The scaling-tier
  galaxies' Einstein radii come from `einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`.
- **Intensities:** The solved-for linear light profile `intensity` values for each MGE Gaussian.
- **Wrap Up:** Summary and next steps.

__Prerequisites__

This script focuses on the API specific to a mixed individually-modelled + scaling-tier extras population. For
background on the underlying single-plane fit API and the MGE source parameterization, you should read first:

 - `autolens_workspace/scripts/imaging/fit.py` — the standard single-plane fit.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the search-based version of this
   script, which composes the same model via `af.Model` with a free `einstein_radius_ref` prior (exponent fixed at 0.5).
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/fit.py` — the MGE source `Basis` API.

All non-linear parameters below are set to the simulator's true values, so the fit visibly recovers the lens
configuration without a search.
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

Load and plot the strong lens dataset `extra_and_scaling_galaxies` via .fits files.
"""
dataset_name = "extra_and_scaling_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

Define a 6.0" circular mask, large enough to include all four extra galaxies (the most distant sits at radius ~5").
"""
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

Load the centres of the individually-modelled extras and the scaling-tier extras. The scaling-tier galaxies need
both centres AND a measured luminosity each, so they're loaded from a CSV via `al.galaxy_table_from_csv`.

In a real analysis, the scaling-tier luminosities come from a prior light-only fit — see
`scripts/group/features/scaling_relation/modeling_for_luminosities.py` for the standalone version of that fit.
"""
individual_extras_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_extras_centres = scaling_table.centres
scaling_extras_luminosities = scaling_table.luminosities

print(f"Individually-modelled extras centres: {list(individual_extras_centres)}")
print(f"Scaling-tier extras centres:          {list(scaling_extras_centres)}")
print(f"Scaling-tier extras luminosities:     {scaling_extras_luminosities}")

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre, so each light profile is evaluated accurately at its peak.
"""
all_galaxy_centres = (
    [(0.0, 0.0)]
    + [tuple(c) for c in individual_extras_centres]
    + [tuple(c) for c in scaling_extras_centres]
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_galaxy_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__MGE Basis__

A `Basis` of 30 linear Gaussians for the source galaxy. The `intensity` of each Gaussian is solved for via linear
algebra at fit time.
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


source_bulge = build_source_basis(centre=(0.0, 0.1))

plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

We compose four populations:

 - `lens` (z=0.5): `SersicSph` light + `IsothermalSph` mass at the origin. The simulator's true Einstein radius
   is 1.6".
 - `individual_extras` (z=0.5): two close companions, each modelled with its own `SersicSph` light +
   `IsothermalSph` mass. Simulator-true Einstein radii: 0.4" and 0.5".
 - `scaling_extras` (z=0.5): two further-out, fainter companions whose Einstein radii are derived from the
   scaling relation. With `einstein_radius_ref=0.2012` at `reference_luminosity=1.0` and per-galaxy luminosity=0.45, each
   acquires `einstein_radius = 0.2012 * (0.45 / 1.0) ** 0.5 = 0.135` — matches the simulator.
 - `source` (z=1.0): the MGE basis above.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=1.5, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6),
)

individual_extras_truth = [
    dict(intensity=0.9, effective_radius=0.6, sersic_index=2.5, einstein_radius=0.4),
    dict(intensity=0.8, effective_radius=0.6, sersic_index=2.5, einstein_radius=0.5),
]

individual_extras = []
for centre, truth in zip(individual_extras_centres, individual_extras_truth):
    individual_extras.append(
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=tuple(centre),
                intensity=truth["intensity"],
                effective_radius=truth["effective_radius"],
                sersic_index=truth["sersic_index"],
            ),
            mass=al.mp.IsothermalSph(
                centre=tuple(centre), einstein_radius=truth["einstein_radius"]
            ),
        )
    )

# reference_luminosity is an explicit fixed constant (Lenstool's "mag0"), not the
# sample max; einstein_radius_ref is the Einstein radius at that reference. Members
# share luminosity 0.45, so einstein_radius_ref * (0.45)**0.5 = 0.135 (simulator truth).
einstein_radius_ref = 0.2012
scaling_exponent = 0.5
reference_luminosity = 1.0

scaling_extras = []
scaling_extras_einstein_radii = []
for centre, luminosity in zip(scaling_extras_centres, scaling_extras_luminosities):
    einstein_radius = (
        einstein_radius_ref * (luminosity / reference_luminosity) ** scaling_exponent
    )
    scaling_extras_einstein_radii.append(einstein_radius)
    scaling_extras.append(
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.SersicSph(
                centre=tuple(centre),
                intensity=luminosity,
                effective_radius=0.5,
                sersic_index=2.5,
            ),
            mass=al.mp.IsothermalSph(
                centre=tuple(centre), einstein_radius=einstein_radius
            ),
        )
    )

source = al.Galaxy(redshift=1.0, bulge=source_bulge)

"""
__Tracer__

The `Tracer` performs the ray-tracing. It queries every mass profile attached to every galaxy in the lens plane
and sums their deflections. For our mixed population, this means the lens's `IsothermalSph`, each individually-
modelled extra's `IsothermalSph`, and each scaling-tier extra's `IsothermalSph` all contribute independent
deflections that sum before mapping image-plane coordinates onto the source-plane.
"""
tracer = al.Tracer(galaxies=[lens] + individual_extras + scaling_extras + [source])

"""
__Fit__

Pass the `Tracer` to a `FitImaging` to fit the dataset.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Scaling Relation Tour__

The scaling-tier galaxies' Einstein radii are NOT free parameters in the model — they're computed from the shared
reference-anchored relation and per-galaxy luminosity. With `einstein_radius_ref=0.2012` at
`reference_luminosity=1.0` and the exponent fixed at 0.5, we have:
"""
for centre, luminosity, er in zip(
    scaling_extras_centres, scaling_extras_luminosities, scaling_extras_einstein_radii
):
    print(
        f"  scaling galaxy @ {tuple(centre)}: luminosity = {luminosity:.3f}, "
        f"einstein_radius = {einstein_radius_ref:.3f} * ({luminosity:.3f} / {reference_luminosity:.3f}) ** {scaling_exponent:.1f} = {er:.4f}"
    )

"""
The lens-plane total deflection is the SUM of every mass profile's contribution. We verify this by computing
each one explicitly and confirming the sum equals what the `Tracer` returns.
"""
grid = dataset.grid

alpha_lens = lens.mass.deflections_yx_2d_from(grid=grid)
alpha_individual = [g.mass.deflections_yx_2d_from(grid=grid) for g in individual_extras]
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_extras]

print(f"\nalpha_lens                (first coord): {alpha_lens[0]}")
print(f"alpha_individual_extra_0 (first coord): {alpha_individual[0][0]}")
print(f"alpha_individual_extra_1 (first coord): {alpha_individual[1][0]}")
print(f"alpha_scaling_extra_0    (first coord): {alpha_scaling[0][0]}")
print(f"alpha_scaling_extra_1    (first coord): {alpha_scaling[1][0]}")

alpha_total_summed = alpha_lens + sum(alpha_individual) + sum(alpha_scaling)

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
The scaling-tier deflections per galaxy are visibly smaller than the lens and individually-modelled extras
because their Einstein radii are an order of magnitude smaller. The shared relation lets us include them in the
model with zero additional free parameters — adding more scaling galaxies would not grow the parameter space.

__Intensities__

After the fit, every linear Gaussian in the source MGE basis has been assigned an `intensity` via linear algebra.
"""
print(
    f"\nFirst Gaussian intensity, source = "
    f"{fit.linear_light_profile_intensity_dict[source_bulge.profile_list[0]]}"
)

tracer_fitted = fit.model_obj_linear_light_profiles_to_light_profiles

subplot_basis_image(basis=tracer_fitted.galaxies[-1].bulge, grid=plot_grid)

"""
__Wrap Up__

This script demonstrated the mixed-strategy API for handling many foreground galaxies — a small number of
individually-modelled extras for the bright, close companions, and an arbitrary number of scaling-tier extras for
the long tail of fainter ones. The scaling relation collapses what would otherwise be N free `einstein_radius`
parameters into 1 shared parameter (`einstein_radius_ref`), keeping the model dimensionality
manageable as galaxy count grows.

In a real modeling workflow:

 - `modeling.py` runs the search-based version, where `einstein_radius_ref` is a free `af.Model`
   parameters with `UniformPrior`s. The luminosities still come from a prior light-only fit.
 - For group-scale lenses with multiple main lens galaxies, see
   `autolens_workspace/scripts/group/features/scaling_relation/` — the three-tier API generalises this script to
   `lens_dict + extra_galaxies + scaling_galaxies` collections.

The key takeaway is that scaling relations let the lens model stay tractable even when 10s or 100s of foreground
galaxies sit on it — the lens-plane deflection is still a simple sum of per-galaxy contributions, but the
contributions from the scaling tier are parameterized through luminosity rather than per-galaxy free parameters.
"""
