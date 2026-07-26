"""
Modeling Features (Multi Galaxy): Scaling Galaxies
==================================================

Extends the multi-galaxy model with a **scaling-galaxies tier**: faint galaxies far from the co-dominant
pair whose masses are tied to their luminosities through a shared relation, so the tier adds ONE free
parameter regardless of how many galaxies it holds.

Two things distinguish this tier at multi-galaxy scale from its group/cluster counterparts, and they are
the point of this example:

 1. **Untruncated profiles are the physically right choice.** Truncation of a galaxy's mass profile
    encodes tidal stripping by a host halo's potential; a multi-galaxy lens has no host halo, so the
    scaling galaxies use plain `IsothermalSph` profiles with

        einstein_radius_i = einstein_radius_ref * (L_i / L_ref) ** 0.5

    (equivalent to sigma ~ L^0.25 since einstein_radius ~ sigma^2). The truncated dPIE variant of this
    tier — physically motivated where a host potential exists — lives in `group/features/group_halo` and
    is the cluster default.

 2. **Expect "a load of galaxies far from the lens."** With no host halo there is no bound member
    population: the tier here is distant, individually-negligible galaxies whose collective contribution
    is a weak (percent-level) correction to the deflection field. It is supported and sometimes
    worthwhile — e.g. wide-field data with many detected neighbours — but unlike at group/cluster scale
    it is NOT a standard ingredient of the model. Fit without it first; add it if residuals or your
    science case (e.g. precision flux ratios) demand it.

The three-tier API (main / extra / scaling galaxies) is identical to the group package's — see
`group/features/scaling_relation/modeling.py` for the fuller API walkthrough and
`imaging/features/extra_galaxies` for the middle (extra) tier, both of which apply here with the single
lens galaxy swapped for the `lens_0`, `lens_1`, ... loop.

__Contents__

- **Dataset & Mask:** Load the scaling-galaxies dataset (auto-simulating if absent).
- **Scaling Catalogue:** Load centres + luminosities from `scaling_galaxies.csv`.
- **Model:** The co-dominant pair (free) + the scaling tier (one shared free parameter).
- **Search + Analysis / Fit / Result.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset & Mask__
"""
dataset_name = "scaling_galaxies"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/scaling_galaxies/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Scaling Catalogue__

The tier's input is the same three-column `y, x, luminosity` CSV schema used at group and cluster scale
(and by the autogalaxy cluster package for light) — one row per scaling galaxy.
"""
scaling_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)

scaling_centres = scaling_table.centres.in_list
scaling_luminosities = scaling_table.luminosities

"""
__Mask__

The mask needs only the lensed emission and the main galaxies — the scaling galaxies sit OUTSIDE it.
Their light never enters the fit; only their mass (deflections reach inside the mask) does. This is
typical of the tier at this scale, and is why their light is not modeled below.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

The co-dominant pair is composed exactly as in `multi_galaxy/modeling.py` (one MGE + `Isothermal` per
galaxy, shear on `lens_0`). The scaling tier then adds one `IsothermalSph` per catalogue row with its
centre fixed and its einstein radius tied to the single shared free `einstein_radius_ref`
(UNTRUNCATED — see the header).
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

# Scaling Galaxies: one shared free parameter for the whole tier.

einstein_radius_ref = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
reference_luminosity = 1.0

scaling_dict = {}

for i, (centre, luminosity) in enumerate(zip(scaling_centres, scaling_luminosities)):
    luminosity_ratio = float(luminosity) / reference_luminosity

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = einstein_radius_ref * luminosity_ratio**0.5

    scaling_dict[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(
    galaxies=af.Collection(**lens_dict, **scaling_dict, source=source)
)

"""
The tier's five galaxies share the single free `einstein_radius_ref` — verify by printing the model.
"""
print(model.info)

"""
__Search + Analysis / Fit__
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="scaling_galaxies",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_full_update=100000,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The truth `einstein_radius_ref` is 0.15" — a real run recovers it (weakly: the tier is a small
perturbation, so expect a broad posterior; that breadth is itself the lesson about how much this tier
matters at multi-galaxy scale).
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

- If your neighbours are close and bright enough to need their own freedom, use the extra-galaxies tier
  (`imaging/features/extra_galaxies`, applied per main galaxy).
- If the galaxies share a dominant halo, you are one rung up the ladder: `group/`, where this same tier
  becomes standard and (in the Lenstool-style workflow) tidally truncated.
"""
