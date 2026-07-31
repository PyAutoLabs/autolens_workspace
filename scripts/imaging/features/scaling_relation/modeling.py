"""
Modeling Features: Scaling Relation
===================================

A strong lens often has many foreground galaxies near the line of sight, and freeing an `einstein_radius` for each
one quickly stops working: with 30 companions the parameter space is too large to sample and the data does not
contain enough information to constrain them individually anyway.

This example ties them to the main lens instead. Each companion's Einstein radius follows a Faber-Jackson relation
**anchored on the main lens galaxy's own Einstein radius**:

    einstein_radius_i = einstein_radius_anchor * (L_i / L_anchor) ** 0.5

`einstein_radius_anchor` is not a new parameter — it is the main lens's `einstein_radius`, which the model is
already fitting. So the entire scaling tier adds **zero free parameters**, however many galaxies it holds. The
`__Two Tiers__` and `__Zero Free Parameters__` sections below are the point of this script.

__Prerequisites__

This script documents only what is specific to the scaling tier. Read these first:

 - `autolens_workspace/scripts/imaging/modeling.py` — the canonical modeling workflow (dataset, mask,
   over-sampling, search, analysis, result).
 - `autolens_workspace/scripts/imaging/features/extra_galaxies/modeling.py` — companions modelled individually,
   which is the tier this script's bounded population belongs to.
 - `autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/modeling.py` — the MGE light API.

__Two Tiers__

Which JSON file a galaxy's centre appears in is what decides how it is modelled:

 - `extra_galaxies_centres.json` -> the **bounded** tier. Each galaxy keeps its own free `einstein_radius`, but
   inside an upper bound derived from its luminosity so it cannot run away to an unphysical mass. Use it for the
   brighter, closer companions that perturb the lensing appreciably on their own.
 - `scaling_galaxies_centres.json` -> the **scaling** tier. Einstein radii are tied to the anchor by the relation
   above. Use it for the long tail of fainter companions, which matter collectively but not individually.

__Untruncated Profiles__

Both tiers use **untruncated** isothermal profiles. Truncation encodes tidal stripping by a host halo's potential,
and a galaxy-scale lens has no host halo. The truncated `dPIEMass` version of this tier is the group- and
cluster-scale default, where a host potential does exist — see `group/features/group_halo` and
`cluster/modeling.py`.

__Relation To The Group And Cluster Packages__

`group/features/scaling_relation` and `cluster/modeling.py` use a *different* normalisation: a standalone free
`einstein_radius_ref`, the Einstein radius of a galaxy at a fixed reference magnitude (Lenstool's `mag0`). That
convention costs one free parameter and is the right one when no single galaxy obviously anchors the system, or
when you want the normalisation to be invariant to which galaxies you place in the tier. Anchoring on the main
lens, as here, costs nothing but does assume the main lens itself sits on the relation.

__Contents__

- **Dataset & Mask:** Load the dataset (auto-simulating if absent) and mask it.
- **Centres:** Three JSON files, one per tier.
- **Luminosities:** The measured luminosities the relation needs, and where they come from.
- **Over Sampling:** Adaptive over-sampling at every galaxy centre.
- **Main Lens & Source:** The anchor and the source.
- **Bounded Tier:** Free Einstein radii inside a luminosity-derived bound.
- **Scaling Tier:** Einstein radii tied to the anchor.
- **Model:** Three top-level collections.
- **Zero Free Parameters:** Proof by parameter count that the tier is free.
- **Search / Analysis / Fit / Result.**
- **CSV Interface:** Loading centres and luminosities from a CSV instead of JSON + Python lists.
- **Wrap Up.**
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset & Mask__

A 6.0" mask, large enough to enclose the scaling tier ~5" out, because this example models their light as well as
their mass.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset", "imaging", dataset_name)

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
__Centres__

One JSON file per tier, each a list of (y, x) arcsecond coordinates. For your own data, the centre-input GUI in
`group/start_here.py` writes these files from mouse clicks.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
bounded_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)
scaling_galaxies_centres = al.from_json(
    file_path=dataset_path / "scaling_galaxies_centres.json"
)

"""
__Luminosities__

The relation needs a measured luminosity for every galaxy it touches, including the anchor. They are given here as
explicit Python lists — the simplest interface, and the one worth reading first. A CSV alternative that scales to
hundreds of galaxies is shown at the end of this script.

**These numbers must be measured; they are not free parameters and they are not guessed.** They come from a
light-only fit performed *before* this one. In this tutorial they are the simulator's truth values, printed when
`simulator.py` runs. In a real analysis, get them from:

 - `scripts/imaging/features/scaling_relation/slam.py` — the SLaM pipeline for this feature. Its light stage fits
   an MGE to every galaxy, integrates each one's Gaussians to a luminosity, then feeds those luminosities into
   exactly the relation below. That is the production path.
 - `scripts/group/features/scaling_relation/modeling_for_luminosities.py` — a standalone light-only fit, if you
   would rather measure luminosities as a separate step.

Only ratios to the anchor enter the relation, so the units do not matter. A magnitude catalogue converts via
`L / L_ref = 10 ** (0.4 * (m_ref - m))`.
"""
luminosity_anchor = 31.0962

bounded_galaxies_luminosities = [3.2595, 2.6076]

scaling_galaxies_luminosities = [1.4939, 1.0865, 0.7696, 0.4980, 0.2716]

assert len(bounded_galaxies_luminosities) == len(list(bounded_galaxies_centres))
assert len(scaling_galaxies_luminosities) == len(list(scaling_galaxies_centres))

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre — anchor, bounded tier and scaling tier alike.
"""
all_centres = (
    list(main_lens_centres)
    + list(bounded_galaxies_centres)
    + list(scaling_galaxies_centres)
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Main Lens & Source__

The main lens is the anchor: an MGE bulge, a free `Isothermal` mass and the system's `ExternalShear`. Its
`einstein_radius` is what the scaling tier hangs off, so it is the one Einstein radius in this model the tier
depends on.
"""
lens_centre = tuple(list(main_lens_centres)[0])

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    centre_prior_is_uniform=True,
    centre=lens_centre,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

"""
__Bounded Tier__

Each galaxy gets an MGE bulge with a fixed centre and an `IsothermalSph` whose `einstein_radius` is free but
bounded above by the Faber-Jackson prediction, doubled to stay conservative:

    upper_limit = min(2 * (einstein_radius_estimate / L_anchor ** 0.5) * L ** 0.5, cap)

Note the asymmetry with the scaling tier below. A prior's `upper_limit` has to be a *number*, so the bound needs an
advance **estimate** of the anchor's Einstein radius — here the Einstein ring's apparent radius, which you can read
straight off the data. The tie in the scaling tier needs no such estimate, because it multiplies the model's own
free parameter rather than a fixed bound. In a SLaM pipeline the estimate is replaced by the previous stage's
fitted value; see `slam.py`.
"""
einstein_radius_estimate = 1.6

einstein_radius_cap = 1.5

bounded_galaxies_list = []

for centre, luminosity in zip(bounded_galaxies_centres, bounded_galaxies_luminosities):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=tuple(centre)
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=min(
            2 * (einstein_radius_estimate / luminosity_anchor**0.5) * luminosity**0.5,
            einstein_radius_cap,
        ),
    )

    bounded_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

extra_galaxies = af.Collection(bounded_galaxies_list)

"""
__Scaling Tier__

The relation itself. `lens.mass.einstein_radius` is the *model's own free parameter*, so multiplying it by the
luminosity ratio produces a derived quantity rather than a new one — this single line is what makes the tier free.

Each galaxy also gets a **spherical** MGE bulge with a fixed centre, which costs no non-linear parameters either:
its Gaussian intensities are solved by linear algebra and its widths are fixed by the basis. So a scaling galaxy is
free in both light and mass. Fitting their light matters here because they sit inside the mask; the multi-galaxy
variant of this feature places the tier outside the mask instead, so its light never enters the fit at all.

The exponent is fixed at the Faber-Jackson value of 0.5 (`einstein_radius ~ sigma^2` and `sigma ~ L^0.25`), which
avoids a normalisation-slope degeneracy. Free it as a systematics test by replacing the constant with
`af.UniformPrior(lower_limit=0.0, upper_limit=1.0)` — that *does* cost one parameter, but still only one for the
whole tier.
"""
scaling_exponent = 0.5

scaling_galaxies_list = []

for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=tuple(centre),
        use_spherical=True,
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = (
        lens.mass.einstein_radius * (luminosity / luminosity_anchor) ** scaling_exponent
    )

    scaling_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
__Model__

Three top-level collections. `scaling_galaxies` is a first-class collection alongside `galaxies` and
`extra_galaxies`: the analysis appends it to the tracer's galaxy list, and the aggregator restores it when results
are loaded back. Keeping the tier in its own collection is therefore not just presentational — it is how the
library expects a scaling population to be expressed, and it keeps `model.info` and `result.info` readable.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Zero Free Parameters__

The claim is worth checking rather than believing. Below, the same model is composed with every scaling galaxy's
`einstein_radius` freed instead of tied, and the parameter counts compared.
"""
scaling_galaxies_free_list = []

for centre in scaling_galaxies_centres:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=tuple(centre),
        use_spherical=True,
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    scaling_galaxies_free_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

model_free = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
    scaling_galaxies=af.Collection(scaling_galaxies_free_list),
)

print(f"\nScaling galaxies in the tier:      {len(scaling_galaxies_list)}")
print(f"Free parameters, tier tied:        {model.prior_count}")
print(f"Free parameters, tier freed:       {model_free.prior_count}")
print(f"Parameters saved by the relation:  {model_free.prior_count - model.prior_count}")

assert model_free.prior_count - model.prior_count == len(scaling_galaxies_list)

"""
__Search / Analysis / Fit__

Standard `Nautilus` + `AnalysisImaging`, exactly as `imaging/modeling.py` describes. The parameter space here is
barely larger than a lens-only fit, which is the whole point: the bounded tier costs 3 parameters per galaxy and
the scaling tier costs nothing at all.
"""
search = af.Nautilus(
    path_prefix=Path("imaging") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

`result.info` lists the three collections separately. The scaling galaxies have no `einstein_radius` entry of their
own — each is reported as a derived function of the lens's, which is what a tied parameter looks like in the
output.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__CSV Interface__

The explicit Python lists above are clear for a handful of galaxies and unwieldy for a hundred. For larger
populations, `al.galaxy_table_from_csv` reads a `y, x, luminosity` CSV (with optional `redshift`) and returns a
`GalaxyTable` with `.centres`, `.luminosities` and `.redshifts`. The simulator writes one per tier, so the block
below is a drop-in replacement for the centre-JSON loads *and* the luminosity lists:

    main_lens_table = al.galaxy_table_from_csv(file_path=dataset_path / "main_lens_galaxies.csv")
    main_lens_centres = main_lens_table.centres
    luminosity_anchor = main_lens_table.luminosities[0]

    bounded_table = al.galaxy_table_from_csv(file_path=dataset_path / "extra_galaxies.csv")
    bounded_galaxies_centres = bounded_table.centres
    bounded_galaxies_luminosities = bounded_table.luminosities

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_galaxies_centres = scaling_table.centres
    scaling_galaxies_luminosities = scaling_table.luminosities

The advantage is a single editable source of truth per tier, with centres and luminosities guaranteed to stay in
the same order — which the two-list interface above cannot enforce for you. Everything downstream is unchanged; the
model composition never sees where the numbers came from.

`slam.py` writes its measured luminosities out in this format, so the CSV interface is the natural one to use once
you have run a light fit on your own data.
"""
main_lens_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "main_lens_galaxies.csv"
)
scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")

print(f"\nAnchor luminosity from CSV:  {main_lens_table.luminosities[0]:.4f}")
print(f"Tier luminosities from CSV:  {list(scaling_table.luminosities)}")

"""
__Wrap Up__

The scaling tier turns N free Einstein radii into zero, by expressing them as a fixed function of a parameter the
model already has. Adding more galaxies to `scaling_galaxies_centres.json` does not grow the model.

Where to go next:

 - `slam.py` in this folder — the production pipeline, which measures the luminosities this script assumed.
 - `fit.py` and `likelihood_function.py` in this folder — the same composition without a search, and the
   per-galaxy deflection sum in detail.
 - `imaging/features/extra_galaxies/modeling.py` — companions given full individual freedom, the tier above this
   one.
 - `multi_galaxy/features/scaling_relation` — the same relation where the anchor is chosen as the brightest of
   several co-dominant deflectors rather than being the only lens.
 - `group/features/scaling_relation` and `cluster/modeling.py` — the reference-magnitude normalisation, and the
   truncated `dPIEMass` profiles appropriate once a host halo exists.
"""
