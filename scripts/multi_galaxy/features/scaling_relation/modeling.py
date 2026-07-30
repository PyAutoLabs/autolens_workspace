"""
Modeling Features (Multi Galaxy): Scaling Relation
=================================================

Extends the multi-galaxy model with a **scaling tier**: faint galaxies far from the co-dominant pair whose Einstein
radii are tied to the brightest of the pair (the BGC) by a Faber-Jackson relation:

    einstein_radius_i = einstein_radius_bgc * (L_i / L_bgc) ** 0.5

`einstein_radius_bgc` is the BGC's own `einstein_radius`, which the model already fits, so the tier adds **zero free
parameters** regardless of how many galaxies it holds.

__What Is Different At Multi-Galaxy Scale__

Three things, and they are the point of this example:

 1. **The anchor has to be identified.** At galaxy scale there is one lens and it is the anchor by default. Here
    there are two co-dominant deflectors and the relation needs the *brightest*. This script picks it by
    `argmax` over the measured luminosities rather than assuming `lens_0` — a distinction that matters because
    which galaxy is listed first in `main_lens_centres.json` is an accident of data preparation, not physics.

 2. **The anchor is tied to, but not tied down.** Both co-dominant galaxies keep free `Isothermal` masses. A
    co-dominant deflector is exactly the kind of galaxy you do not want constrained by a scaling law — the whole
    reason this regime exists is that each one contributes comparably and deserves its own freedom. Only the faint
    tier is tied.

 3. **Untruncated profiles are the physically right choice, not a simplification.** Truncation encodes tidal
    stripping by a host halo's potential, and a multi-galaxy lens has no host halo. The truncated `dPIEMass`
    variant of this tier — physically motivated where a host potential does exist — lives in
    `group/features/group_halo` and is the cluster-scale default.

__Expect "A Load Of Galaxies Far From The Lens"__

With no host halo there is no bound member population: the tier here is distant, individually-minor galaxies whose
collective contribution is a weak (percent-level shear) correction to the deflection field. It is supported and
sometimes worthwhile — wide-field data with many detected neighbours, or a science case like precision flux
ratios — but unlike at group/cluster scale it is NOT a standard ingredient of the model. Fit without it first; add
it if residuals or your science case demand it.

__Prerequisites__

This script documents only what is specific to the scaling tier at this scale. Read first:

 - `autolens_workspace/scripts/multi_galaxy/modeling.py` — the co-dominant-pair model this extends.
 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the fuller walkthrough of the
   relation itself, with the single-lens anchor.

__Mass-Only Tier__

The mask needs only the lensed emission and the co-dominant pair — the scaling galaxies sit OUTSIDE it, 5.5-7" out.
Their light therefore never enters this fit; only their mass, whose deflections reach inside the mask. This is
typical of the tier at this scale and is why no light is modelled for them below. `slam.py` takes the opposite
approach on a deliberately enlarged mask, because it has to *measure* their luminosities.

__Contents__

- **Dataset & Mask:** Load the dataset (auto-simulating if absent) and mask the pair.
- **Centres:** The pair's centres and the tier's centres.
- **Luminosities:** The measured luminosities, and identifying the BGC from them.
- **Over Sampling / Main Lens Galaxies / Source.**
- **Scaling Tier:** Einstein radii tied to the BGC.
- **Model:** Two top-level collections.
- **Zero Free Parameters:** Proof by parameter count.
- **Search / Analysis / Fit / Result.**
- **CSV Interface:** The same inputs from a CSV.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import numpy as np
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset & Mask__

A 3.0" mask, as in `multi_galaxy/modeling.py`. The scaling galaxies lie well outside it.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

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
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

"""
__Luminosities + The BGC__

Explicit Python lists — the simplest interface, and the one worth reading first; the CSV equivalent is at the end.

**These luminosities must be measured; they are not free parameters and they are not guessed.** They come from a
light-only fit performed before this one. In this tutorial they are the simulator's truth values, printed when
`simulator.py` runs. On real data use `slam.py` in this folder, whose light stages fit an MGE to every galaxy — the
pair on the standard mask and the distant tier on an enlarged one — and integrate each to a luminosity.

The BGC is then whichever main lens is brightest. Note this is a *measurement*, not a naming convention: if
`main_lens_centres.json` listed the fainter galaxy first, `bgc_index` would simply come out as 1 and everything
downstream would still anchor on the right galaxy.
"""
main_lens_luminosities = [9.7913, 5.6663]

scaling_galaxies_luminosities = [1.2636, 0.8845, 0.6318, 0.3791, 0.2527]

assert len(main_lens_luminosities) == len(list(main_lens_centres))
assert len(scaling_galaxies_luminosities) == len(list(scaling_galaxies_centres))

bgc_index = int(np.argmax(main_lens_luminosities))
luminosity_bgc = main_lens_luminosities[bgc_index]
bgc_key = f"lens_{bgc_index}"

print(f"BGC is {bgc_key}, L_bgc = {luminosity_bgc}")

"""
__Over Sampling__

Centred on the co-dominant pair only — the tier is outside the mask, so there is nothing of theirs to over-sample.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Galaxies__

One MGE bulge + free `Isothermal` mass per co-dominant deflector, exactly as `multi_galaxy/modeling.py` composes
them. The BGC's `einstein_radius` is what the tier hangs off, but it is otherwise an ordinary free parameter.
"""
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

"""
__Source__
"""
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

"""
__Scaling Tier__

The relation. `lens_dict[bgc_key].mass.einstein_radius` is the BGC's free parameter, so multiplying it by each
member's luminosity ratio produces a derived quantity rather than a new one.

Mass only: no `bulge` is given, because these galaxies sit outside the mask and their light is not in the fit.

The exponent is fixed at the Faber-Jackson value of 0.5 (`einstein_radius ~ sigma^2`, `sigma ~ L^0.25`).
"""
scaling_exponent = 0.5

scaling_galaxies_list = []

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = (
        lens_dict[bgc_key].mass.einstein_radius
        * (luminosity / luminosity_bgc) ** scaling_exponent
    )

    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass))

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
__Model__

Two top-level collections. `scaling_galaxies` is a first-class collection alongside `galaxies`: the analysis appends
it to the tracer's galaxy list and the aggregator restores it when results are loaded back, so the tier belongs
there rather than folded in among the main lenses. Keeping it separate also keeps `model.info` readable — the
co-dominant pair and the tied population are visibly different populations.
"""
model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Zero Free Parameters__

Worth checking rather than believing: the same model with every member's `einstein_radius` freed instead of tied.
"""
scaling_galaxies_free_list = []

for centre in scaling_galaxies_centres:
    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    scaling_galaxies_free_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass))

model_free = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    scaling_galaxies=af.Collection(scaling_galaxies_free_list),
)

print(f"\nScaling galaxies in the tier:      {len(scaling_galaxies_list)}")
print(f"Free parameters, tier tied:        {model.prior_count}")
print(f"Free parameters, tier freed:       {model_free.prior_count}")
print(f"Parameters saved by the relation:  {model_free.prior_count - model.prior_count}")

assert model_free.prior_count - model.prior_count == len(scaling_galaxies_list)

"""
__Search / Analysis / Fit__
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_full_update=100000,
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

Expect the tier to be *weakly* constrained, and expect that to show up as a broad posterior on the BGC's Einstein
radius rather than on the members themselves — they have no parameters of their own. The tier is a percent-level
perturbation on a system dominated by two co-dominant deflectors, so the data does not push back hard on it. That
breadth is itself the lesson about how much this tier matters at multi-galaxy scale.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__CSV Interface__

The explicit lists above are clear for a handful of galaxies and unwieldy for a hundred. `al.galaxy_table_from_csv`
reads a `y, x, luminosity` CSV and returns a `GalaxyTable` with `.centres`, `.luminosities` and `.redshifts`,
keeping centres and luminosities in one file that cannot fall out of order:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    main_lens_centres = main_lens_table.centres
    main_lens_luminosities = main_lens_table.luminosities

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

The `argmax` that identifies the BGC then runs on the CSV's luminosities, unchanged.
"""
main_lens_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "main_lens_galaxies.csv"
)

print(f"\nMain lens luminosities from CSV: {list(main_lens_table.luminosities)}")
print(f"BGC index from CSV:              {int(np.argmax(main_lens_table.luminosities))}")

"""
__Wrap Up__

- If a companion is bright and close enough to need its own freedom, it belongs in the extra-galaxies tier
  (`multi_galaxy/features/extra_galaxies` in the AutoGalaxy workspace, or `imaging/features/extra_galaxies`
  applied per main galaxy).
- If the galaxies share a dominant halo, you are one rung up the ladder: `group/`, where this tier becomes standard
  and — in the Lenstool-style workflow — tidally truncated `dPIEMass`.
- `slam.py` in this folder measures the luminosities this script assumed, using two masks to reach the distant tier.
"""
