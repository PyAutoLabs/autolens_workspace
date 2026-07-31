"""
Modeling Features (Group): Scaling Relations
============================================

Group-scale strong lenses can have many galaxies in the foreground beyond the primary lens. As the number grows,
modelling each galaxy individually becomes impractical: a system with 10 companions would gain 10 extra mass
free parameters, and the data is rarely informative enough to constrain them all.

This example demonstrates the **three-tier modeling API** used by the production group pipelines, in which foreground
galaxies are split into three distinct populations:

 - **Main lens galaxies** (`main_lens_centres.json`): the primary lens(es). Modelled with an MGE bulge + free
   `Isothermal` mass + `ExternalShear` (on `lens_0` only). These dominate the lensing.

 - **Extra galaxies** (`extra_galaxies_centres.json`): nearby companion galaxies modelled individually, each with its
   own MGE bulge and a tidally truncated `dPIEMassSph` mass with bounded free `sigma`. Their light is fit and their
   mass is fit but constrained to a sensible range. Use this tier for the brighter / closer companions that
   contribute non-trivially to the lensing on their own.

 - **Scaling galaxies** (`scaling_galaxies_centres.json`): further-out, fainter companions whose truncated
   `dPIEMassSph` masses are tied together via a shared scaling relation in Lenstool's native parameters:

       sigma = sigma_ref * (luminosity / reference_luminosity) ** 0.25
       r_cut = r_cut_ref * (luminosity / reference_luminosity) ** 0.7

   anchored to a fixed *reference magnitude* (Lenstool's ``mag0``, an explicit constant — not the sample max), with
   the exponents fixed at the modern tied values (sigma ∝ L^0.25 is Faber-Jackson; beta_cut = 1 + gamma - 2*alpha
   with gamma = 0.2) — the convention used by Lenstool and standard in published group- and cluster-scale analyses
   (Bergamini et al. 2019). The only free parameter is `sigma_ref` — adding more scaling galaxies does not grow
   the model. Use this tier for the long tail of fainter companions.

Both tiers use the tidally truncated dPIE profile, the group and cluster convention: members orbiting in the shared
group potential have their outer dark matter stripped by tides, so their profiles are physically truncated at
``r_cut`` (at galaxy scale, with no host environment, extra galaxies are untruncated — see
`imaging/features/extra_galaxies`).

Splitting galaxies across these three tiers is the standard pattern in production group fits (see the `euclid_strong_lens_modeling_pipeline` repository). It gives the lensing-significant galaxies the model flexibility they need
while keeping the model tractable as the number of foreground galaxies grows.

For the galaxy-scale counterpart — one main lens, one bounded tier and one scaling tier — see
`autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py`. Note it uses a **different
normalisation**: it anchors the relation on the main lens galaxy's own `einstein_radius`, so its scaling tier costs
zero free parameters. That is attractive but assumes one galaxy obviously anchors the system and that it sits on the
relation itself. This example instead uses the reference-magnitude normalisation (Lenstool's ``mag0``), which costs
one free parameter and stays invariant to which galaxies you place in the tier — the right choice at group scale,
where the BGG is not always dominant and the members share a host halo.

__Contents__

- **Three-Tier API:** Why split foreground galaxies into main, extra and scaling tiers.
- **Centres:** Three JSON files, one per tier, loaded with `al.from_json`.
- **Luminosities:** The scaling galaxies need a measured luminosity each; in this tutorial we hardcode them.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Main Lens Galaxies:** MGE bulge + free `Isothermal` mass; `ExternalShear` only on `lens_0`.
- **Extra Galaxies:** MGE bulge with fixed centre + truncated `dPIEMassSph` with bounded uniform `sigma`.
- **Scaling Galaxies:** MGE bulge with fixed centre + truncated `dPIEMassSph` mass via shared scaling relation.
- **Model:** Compose the lens model fitted to the data.
- **Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
- **Search and Analysis:** Configure the non-linear search and run the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Three-Tier API__

The three-tier split is the load-bearing idea here. To make the right choice for a given galaxy, ask:

 - Is it bright enough that fitting its light independently meaningfully helps the lens model? -> main or extra tier.
 - Does it dominate the lensing? -> main tier.
 - Is it close enough / bright enough to need its own free mass normalization? -> extra tier.
 - Is it part of the long tail of fainter companions, where individually it contributes little but collectively it
   matters? -> scaling tier.

In this example we have one main galaxy, two extras, and two scaling galaxies, but the same code scales naturally to
many more on each tier — the JSON centre files and the per-galaxy loops are the only things that grow.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

This example uses its own dataset under `dataset/group/scaling_relation/`, simulated by the paired simulator at
`scripts/group/features/scaling_relation/simulator.py`. The dataset has three centre JSON files — one per tier — so
we exercise the full three-tier API.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset", "group", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist, run the paired simulator first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/features/scaling_relation/simulator.py"],
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

We use a slightly larger mask radius than `group/modeling.py` (8.5") to enclose the scaling galaxies, which are placed
further out from the lens than the extras.
"""
mask_radius = 8.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres__

Centres for the main and extra tiers come from JSON files (one (y, x) tuple per galaxy each). The scaling tier loads
its centres AND luminosities from a single CSV via `al.galaxy_table_from_csv` — see the next section.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

print(f"Main lens centres: {main_lens_centres}")
print(f"Extra galaxies centres: {extra_galaxies_centres}")

"""
__Scaling Galaxy Centres + Luminosities__

The scaling relation needs both centres AND a measured luminosity per scaling galaxy. There are two equally-supported
ways to provide them in PyAutoLens — both shown below so you can pick whichever fits your workflow.

**Option A — CSV via `al.galaxy_table_from_csv` (recommended for non-trivial galaxy counts).** The simulator writes a
`scaling_galaxies.csv` with columns `y, x, luminosity` (and optional `redshift`) alongside the centre JSONs. We load it
in one call which returns a typed `GalaxyTable` with `.centres` (a `Grid2DIrregular`), `.luminosities`, and (optionally)
`.redshifts`. This scales naturally to populations of tens or hundreds of galaxies — the source of truth lives in a
single editable file.

**Option B — JSON centres + hardcoded luminosity list (the original API, fine for short, fixed-length tutorials).**
Load the centres from `scaling_galaxies_centres.json` with `al.from_json` (the same loader used for the main and
extras tiers above) and define the luminosities as a Python list. Concise and obvious for small populations; awkward
once you have more than a handful.

In a real analysis the luminosities come from a prior light-only fit. Two production patterns for obtaining them:

 - **Standalone light-only fit.** Run a single-stage non-linear search whose model is just MGE bulges for every galaxy
   (no mass, no source). After the fit, compute total luminosity per galaxy from the bulge gaussian parameters:
   `total_luminosity = sum(2 * pi * sigma**2 / axis_ratio * intensity) / pixel_scale**2`. The standalone example at
   `scripts/group/features/scaling_relation/modeling_for_luminosities.py` writes its result as a `scaling_galaxies.csv`
   in the dataset folder, which can then be loaded directly via Option A here.

 - **As the `source_lp[0]` stage of a SLaM pipeline.** Every group SLaM script defines a `source_lp_0` function whose
   sole purpose is to fit a light-only MGE model to the main lens, extra galaxies and scaling galaxies in one go.
   Subsequent stages chain from this result to compute luminosities and bound / scale the per-galaxy mass models.
   See:

       scripts/group/slam.py                            (the canonical SLaM pipeline for group lenses)
       scripts/group/features/pixelization/slam.py      (pixelization variant)

   Search for `source_lp_0(` in either file — the pattern is identical and is documented in the function's header
   docstring. The luminosity computation lives in the `source_lp_1` function that consumes the `source_lp_result_0`
   result.

We use Option A by default below. The Option B equivalent is shown commented out — uncomment it (and comment out
Option A) to switch.
"""
# Option A: CSV (recommended)
scaling_galaxies_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_galaxies_centres = scaling_galaxies_table.centres
scaling_galaxies_luminosity_list = scaling_galaxies_table.luminosities

# Option B: JSON centres + hardcoded luminosities (uncomment to use instead)
# scaling_galaxies_centres = al.from_json(
#     file_path=dataset_path / "scaling_galaxies_centres.json"
# )
# scaling_galaxies_luminosity_list = [0.45, 0.45]
# assert len(scaling_galaxies_luminosity_list) == len(list(scaling_galaxies_centres)), (
#     "Number of scaling-galaxy luminosities must match the number of scaling-galaxy centres."
# )

print(f"Scaling galaxies centres: {scaling_galaxies_centres}")
print(f"Scaling galaxies luminosities: {scaling_galaxies_luminosity_list}")

"""
__Main Lens Galaxies__

One MGE bulge + free `Isothermal` mass per main lens; `ExternalShear` only on `lens_0`. Mirrors `group/modeling.py`.
"""
lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

"""
__Extra Galaxies__

Each modelled with its own MGE bulge (fixed centre) + tidally truncated `dPIEMassSph` mass with a bounded uniform
`sigma` (`r_core` and `r_cut` fixed; the redshifts and cosmology constants are pinned so they do not inherit
default priors and float). Each adds 1 free sigma parameter to the model.
"""
extra_galaxies_list = []

for centre in extra_galaxies_centres:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=tuple(centre)
    )

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=350.0)
    mass.r_core = 0.0  # vanishing core — fixed; the dPIE is analytic at r_core = 0
    mass.r_cut = 10.0  # truncation fixed at a fiducial radius
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66  # pinned: model constants, not parameters to sample
    mass.Om0 = 0.30966

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

"""
__Scaling Galaxies__

The scaling-relation tier, in the reference-anchored convention used by Lenstool and essentially every published
group- and cluster-scale analysis (Limousin et al. 2005; Eliasdottir et al. 2007; Bergamini et al. 2019), expressed
directly in Lenstool's native dPIE parameters: ``sigma ∝ L^0.25`` (Faber-Jackson — equivalent to
einstein_radius ∝ L^0.5 since the lens strength goes as sigma²) and ``r_cut ∝ L^0.7`` via the modern tied exponent
beta_cut = 1 + gamma - 2*alpha with gamma = 0.2, with vanishing unscaled cores. The normalization ``sigma_ref`` is
the velocity dispersion of a galaxy *at the reference magnitude* — a physically interpretable quantity with an
easy-to-motivate prior range — and it is defined ONCE outside the loop: every scaling galaxy's mass derives from it
via its luminosity ratio to the reference. The reference luminosity ``reference_luminosity`` is an **explicit fixed
constant** (Lenstool's reference magnitude ``mag0``), *not* the maximum luminosity of the sample — anchoring to a
fixed reference keeps the normalization invariant to which galaxies are placed in the tier. In a real analysis set
it to the BCG/BGG magnitude (or a characteristic L*); here we use a fiducial ``reference_luminosity = 1.0``. The
exponents are *fixed* rather than fitted, avoiding the normalization-slope degeneracy. Only luminosity ratios
enter, so the luminosity units are irrelevant; magnitude catalogues convert via
``L / L_ref = 10 ** (0.4 * (m_ref - m))``.

The same relation at cluster scale — with hundreds of members — is ``scripts/cluster/modeling.py``. To free an
exponent as a systematics test, replace its fixed value with e.g. ``af.UniformPrior(lower_limit=0.0, upper_limit=0.5)``.

Adding more scaling galaxies (e.g. by lengthening the centres + luminosity lists) does not add any free parameters
to the model.
"""
sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=200.0)
scaling_sigma_exponent = 0.25  # alpha (Faber-Jackson)
scaling_gamma = 0.2  # M/L tilt, universally fixed
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7
scaling_r_cut_ref = 5.0  # fixed reference truncation radius

reference_luminosity = 1.0

scaling_galaxies_list = []

for scaling_galaxy_centre, scaling_galaxy_luminosity in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosity_list
):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=tuple(scaling_galaxy_centre),
    )

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(scaling_galaxy_centre)
    luminosity_ratio = scaling_galaxy_luminosity / reference_luminosity
    mass.sigma = sigma_ref * luminosity_ratio**scaling_sigma_exponent
    mass.r_core = 0.0  # vanishing core — fixed, never scaled
    mass.r_cut = scaling_r_cut_ref * luminosity_ratio**scaling_rcut_exponent
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66
    mass.Om0 = 0.30966

    scaling_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    scaling_galaxies_list.append(scaling_galaxy)

scaling_galaxies = af.Collection(scaling_galaxies_list)

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
__Model__

Each tier sits in its own top-level collection. This makes `model.info` and `result.info` easy to read — main lenses
appear under `galaxies`, individually-modelled companions under `extra_galaxies`, and the scaling-relation tier under
`scaling_galaxies`.
"""
model = af.Collection(
    galaxies=af.Collection(**lens_dict, source=source),
    extra_galaxies=extra_galaxies,
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre — main, extras and scaling alike.
"""
all_centres = (
    list(main_lens_centres)
    + list(extra_galaxies_centres)
    + list(scaling_galaxies_centres)
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("group") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

`result.info` shows all three tiers separately. The recovered `sigma_ref` should be close to the truth
value used by the simulator (106.0 km/s, the velocity dispersion of a reference-magnitude galaxy at
`reference_luminosity = 1.0`; each member, at luminosity 0.45, then has sigma = 86.8 km/s, r_cut = 2.86" and an
Einstein radius of ~0.136").
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

This example showed the full three-tier modeling API. The same structure scales naturally to systems with many more
foreground galaxies — the only thing that grows is the JSON centre files. The relation can also be applied to other
mass profiles or other measured quantities by swapping the `dPIEMassSph` for any other `MassProfile` or the
luminosity for stellar mass / velocity dispersion.

Related examples:

 - `autolens_workspace/scripts/imaging/features/scaling_relation/modeling.py` — the galaxy-scale counterpart, which
   anchors the relation on the main lens's own `einstein_radius` (zero free parameters) instead of a reference
   magnitude.
 - `autolens_workspace/scripts/multi_galaxy/features/scaling_relation/modeling.py` — the same anchored relation with
   the anchor chosen as the brightest of several co-dominant deflectors.
 - `autolens_workspace/scripts/group/features/scaling_relation/modeling_for_luminosities.py` — the standalone
   light-only fit that produces the `scaling_galaxies_luminosity_list` used here.
 - `autolens_workspace/scripts/group/slam.py` and friends — the SLaM pipeline equivalent (`source_lp[0]` stage).
"""
