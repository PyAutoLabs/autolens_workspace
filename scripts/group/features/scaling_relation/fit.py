"""
Features: Group Scaling Relation Fit
====================================

A group-scale strong lens often has many foreground galaxies near the line of sight to the source, on top of one
or more primary lens galaxies. The **three-tier API** splits these galaxies into populations that the lens model
treats differently:

 - **Main lens galaxies** (`main_lens_centres.json`): the primary lens(es) — each modelled with its own free
   mass parameters via the group `lens_dict` API.
 - **Extra galaxies** (`extra_galaxies_centres.json`): individually-modelled companions, each with its own free
   `einstein_radius`. Use this tier for the brighter / closer companions that contribute non-trivially to the
   lensing on their own.
 - **Scaling galaxies** (`scaling_galaxies_centres.json` + `scaling_galaxies.csv`): the long tail of fainter
   companions whose Einstein radii are tied together via a shared reference-anchored relation
   `einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5` (exponent fixed at the
   Faber-Jackson value; the Lenstool convention). Adding more galaxies to this tier does not grow the model.

This script illustrates the API for performing a fit to a group-scale strong lens with all three tiers active,
via the standard `Tracer` and `FitImaging` objects, without invoking a non-linear search.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Centres + Luminosities:** Load main, extras and scaling-tier centres + luminosities.
- **Over Sampling:** Adaptive over-sampling at every galaxy centre.
- **MGE Basis:** Build a `Basis` of linear Gaussians for the source.
- **Galaxies:** Concrete composition — `lens_dict` + extras + scaling-tier + source.
- **Tracer:** Build the `Tracer` and fit the dataset.
- **Three-Tier Deflection Tour:** Per-tier deflection sums into the tracer's total deflection.
- **Intensities:** The solved-for linear light profile `intensity` values for each MGE Gaussian.
- **Wrap Up:** Summary and next steps.

__Prerequisites__

This script focuses on the API specific to a group-scale three-tier extras population. For background:

 - `autolens_workspace/scripts/imaging/features/scaling_relation/fit.py` — the single-main-lens version of this
   script. The three-tier walkthrough below generalises that example across multiple main lens galaxies. It anchors
   its relation on the main lens's own Einstein radius rather than on a reference magnitude, so the two scripts
   compose the tier from different normalisations; the deflection-sum machinery is identical.
 - `autolens_workspace/scripts/group/start_here.py` — the group-scale `lens_dict` API, including how
   `main_lens_centres.json` is loaded.
 - `autolens_workspace/scripts/group/features/scaling_relation/modeling.py` — the search-based version of this
   script, which composes the same model via `af.Model` with a free `einstein_radius_ref` prior.

The group simulator here has only ONE main lens galaxy, so the `lens_dict` has a single entry `lens_0`. The
pattern generalises naturally to groups with multiple main lens galaxies.
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

Load and plot the group-scale strong lens dataset `scaling_relation` via .fits files.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "group" / dataset_name

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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres + Luminosities__

Load each tier's data:
 - main lens centres from JSON,
 - individually-modelled extras centres from JSON,
 - scaling-tier centres AND luminosities from CSV.
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

print(f"Main lens centres:                  {list(main_lens_centres)}")
print(f"Individually-modelled extras centres: {list(extra_galaxies_centres)}")
print(f"Scaling-tier extras centres:          {list(scaling_galaxies_centres)}")
print(f"Scaling-tier extras luminosities:     {scaling_galaxies_luminosities}")

"""
__Mask__

Define an 8.0" circular mask, large enough to include the main lens and all extra + scaling galaxies (the most
distant sits at radius ~7.5").
"""
mask_radius = 8.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre across all three tiers.
"""
all_galaxy_centres = (
    [tuple(c) for c in main_lens_centres]
    + [tuple(c) for c in extra_galaxies_centres]
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
__MGE Basis__

A `Basis` of 30 linear Gaussians for the source galaxy.
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

Three-tier concrete composition:

 - `lens_dict` (z=0.5): one `Galaxy` per main lens centre, each with `SersicSph` light + `IsothermalSph` mass.
   The simulator here has a single main lens with `einstein_radius=4.0`.
 - `individual_extras` (z=0.5): two individually-modelled companions with simulator-true Einstein radii 0.8
   and 1.0.
 - `scaling_extras` (z=0.5): two scaling-tier companions whose Einstein radii are derived from
   `einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5` (simulator truth:
   0.135 each from luminosity 0.45).
 - `source` (z=1.0): the MGE basis above.
"""
main_lens_truth = [
    dict(intensity=0.7, effective_radius=2.0, sersic_index=4.0, einstein_radius=4.0),
]

lens_dict = {}
for i, (centre, truth) in enumerate(zip(main_lens_centres, main_lens_truth)):
    lens_dict[f"lens_{i}"] = al.Galaxy(
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

extra_truth = [
    dict(intensity=0.9, effective_radius=0.8, sersic_index=3.0, einstein_radius=0.8),
    dict(intensity=0.9, effective_radius=0.8, sersic_index=3.0, einstein_radius=1.0),
]

individual_extras = []
for centre, truth in zip(extra_galaxies_centres, extra_truth):
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

# reference_luminosity is an explicit fixed constant (Lenstool's reference
# magnitude "mag0"), not the sample max; einstein_radius_ref is the Einstein
# radius of a galaxy at that reference. Here L_ref = 1.0 (fiducial); both members
# share luminosity 0.45, so einstein_radius_ref * (0.45)**0.5 = 0.135 (simulator truth).
einstein_radius_ref = 0.2012
scaling_exponent = 0.5
reference_luminosity = 1.0

scaling_extras = []
scaling_extras_einstein_radii = []
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
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
                effective_radius=0.6,
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

The `Tracer` queries every mass profile across all three tiers and sums their deflections.
"""
tracer = al.Tracer(
    galaxies=list(lens_dict.values()) + individual_extras + scaling_extras + [source]
)

"""
__Fit__
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Three-Tier Deflection Tour__

The lens-plane total deflection is the sum of three tier-wise contributions:

  alpha_lens_total(theta) = sum_i alpha_lens_i(theta)
                          + sum_j alpha_extra_individual_j(theta)
                          + sum_k alpha_extra_scaling_k(theta)

where the scaling tier contributions come from mass profiles whose Einstein radii are derived from the shared
relation. We verify this by computing each tier explicitly and confirming the grand sum equals what the `Tracer`
returns.
"""
grid = dataset.grid

alpha_main_per_lens = [
    g.mass.deflections_yx_2d_from(grid=grid) for g in lens_dict.values()
]
alpha_individual = [g.mass.deflections_yx_2d_from(grid=grid) for g in individual_extras]
alpha_scaling = [g.mass.deflections_yx_2d_from(grid=grid) for g in scaling_extras]

alpha_main_total = sum(alpha_main_per_lens)
alpha_individual_total = sum(alpha_individual)
alpha_scaling_total = sum(alpha_scaling)

print(f"alpha_main_lens (tier sum, first coord)  : {alpha_main_total[0]}")
print(f"alpha_individual (tier sum, first coord) : {alpha_individual_total[0]}")
print(f"alpha_scaling    (tier sum, first coord) : {alpha_scaling_total[0]}")

for centre, luminosity, er in zip(
    scaling_galaxies_centres,
    scaling_galaxies_luminosities,
    scaling_extras_einstein_radii,
):
    print(
        f"    scaling galaxy @ {tuple(centre)}: "
        f"einstein_radius = {einstein_radius_ref:.3f} * ({luminosity:.3f} / {reference_luminosity:.3f}) ** {scaling_exponent:.1f} = {er:.4f}"
    )

alpha_total_summed = alpha_main_total + alpha_individual_total + alpha_scaling_total

traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
alpha_total_tracer = grid - traced_grids[1]

print(f"\nalpha_total (summed by hand, first 3): {alpha_total_summed[:3]}")
print(f"alpha_total (from tracer,    first 3): {alpha_total_tracer[:3]}")

assert np.allclose(np.asarray(alpha_total_summed), np.asarray(alpha_total_tracer))

"""
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

This script demonstrated the group-scale three-tier API and the per-tier deflection composition, without
invoking a non-linear search. The scaling relation collapses what would otherwise be N free `einstein_radius`
parameters into a single shared normalization (`einstein_radius_ref`, the Einstein radius of a reference-magnitude galaxy,
with the exponent fixed at 0.5), letting the model dimensionality stay constant as galaxy count grows.

In a real modeling workflow:

 - `modeling.py` runs the search-based version, where `einstein_radius_ref` is a free `af.Model` parameter with
   a `UniformPrior`.
 - `modeling_for_luminosities.py` is the standalone light-only fit that produces the luminosities consumed by
   the scaling relation. In production this stage is the `source_lp[0]` step of a SLaM pipeline.
 - `autolens_workspace/scripts/group/slam.py` is the full SLaM pipeline, which already implements scaling
   galaxies via this same composition under the hood.

The key takeaway is that the group `lens_dict + extra_galaxies + scaling_galaxies` collection structure scales
naturally to any number of main lens galaxies and any number of foreground galaxies — the lens-plane deflection
is still a simple per-galaxy sum, but the scaling-tier contributions are parameterized through luminosity rather
than per-galaxy free parameters.
"""
