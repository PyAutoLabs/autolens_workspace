"""
Modeling Features: Scaling Relations
====================================

Strong lenses often have many galaxies surrounding the lens galaxy whose mass contributes to the ray-tracing of the
source. The `extra_galaxies` example shows how to add each of these galaxies into the model with their own light and
mass profiles, fixing their centres to the observed centres of light. That works well when only a handful of extra
galaxies are present, but rapidly becomes unwieldy as the number grows. With 10 extra galaxies modelled with
`IsothermalSph` mass profiles, the lens model gains 10 additional `einstein_radius` free parameters; with 30, the
parameter space is so large that the non-linear search struggles to converge and the data lacks the information content
to constrain every galaxy individually.

A common solution is to model the lensing contribution of these galaxies via a **scaling relation**. An easier-to-measure
property of each galaxy (typically luminosity, but it could also be stellar mass or velocity dispersion) is related to
its mass profile via a shared, reference-anchored relation (the Lenstool convention):

    einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5

The single free parameter is `einstein_radius_ref` — the Einstein radius of a galaxy at a fixed reference magnitude
(Lenstool's `mag0`) — regardless of how many galaxies sit on the relation; the exponent is fixed at the Faber-Jackson
value of 0.5. The luminosities act as priors on the masses, ensuring each galaxy's contribution stays
physically reasonable.

This example demonstrates the **mixed-strategy** pattern: a single `extra_galaxies` collection that contains BOTH
galaxies modelled individually (each with its own free Einstein radius) AND galaxies on a shared scaling relation. This
is the typical real-world configuration: the brighter / closer companions get individual mass parameters because they
contribute non-trivially to the lensing on their own, while the long tail of fainter companions sit on the relation.

The dataset used here is `dataset/imaging/extra_and_scaling_galaxies`, simulated by the paired script
`scripts/imaging/features/scaling_relation/simulator.py`. It contains a galaxy-scale lens at the origin, two close
companions (the "individual" tier here), and two fainter further-out companions (the "scaling-relation" tier). All four
companions live in the same `extra_galaxies` collection in this imaging-context example — the terminology
`scaling_galaxies` for a separate top-level collection is reserved for the group-scale example.

__Contents__

- **Two Strategies, One Collection:** Why mix individual + relational extras in the same `extra_galaxies` collection.
- **Centres:** Two JSON files load the centres of the individually-modelled extras and the scaling-relation extras.
- **Luminosities:** The scaling-relation tier needs a measured luminosity per galaxy.
- **Where do luminosities come from?:** The `modeling_for_luminosities.py` example and the SLAM `source_lp[0]` step.
- **Redshifts:** All foreground galaxies are at the same redshift as the lens galaxy.
- **Group vs Imaging:** Where the group-scale variant lives.
- **Dataset & Mask:** Standard set up of the dataset and mask.
- **Lens & Source:** MGE bulge + Isothermal mass + ExternalShear lens; MGE source.
- **Individually-Modelled Extras:** Bounded `UniformPrior` on `einstein_radius` per galaxy.
- **Scaling-Relation Extras:** A single shared `einstein_radius_ref` prior (exponent fixed at 0.5).
- **Model:** Compose the lens model fitted to the data.
- **Over Sampling:** Adaptive over-sampling at every galaxy centre.
- **Search and Analysis:** Configure the non-linear search and run the model-fit.
- **Wrap Up:** Summary of the script and next steps.

__Two Strategies, One Collection__

There is no architectural distinction in PyAutoLens between "individual" and "scaling-relation" extra galaxies — both
sit in the same `extra_galaxies = af.Collection([...])`. The distinction is purely in how the per-galaxy `Galaxy` model
is built:

  - For an individually-modelled galaxy: `mass.einstein_radius = af.UniformPrior(...)` — one free parameter per galaxy.
  - For a scaling-relation galaxy: `mass.einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`
    — zero new free parameters per galaxy, because the single shared normalization is used across the whole tier.

The two strategies coexist freely in the same collection. This script builds a Python list, pushes the individually-
modelled galaxies onto it first, then the relational galaxies, and wraps the whole thing in a single `af.Collection`.

__Where do luminosities come from?__

In a real analysis the luminosities used by the scaling relation are not known a priori — they have to be measured
from the data itself. Two production patterns:

 - **Standalone light-only fit.** Run a single non-linear search whose model is just MGE bulges for every galaxy
   (no mass, no source). After the fit, compute total luminosity per galaxy from the bulge gaussian intensities:
   `total_luminosity = sum(2 * pi * sigma**2 / axis_ratio * intensity) / pixel_scale**2`. Feed those numbers into the
   scaling-relation model below. See `scripts/group/features/scaling_relation/modeling_for_luminosities.py` for a
   worked example.

 - **As the first stage of a SLaM pipeline.** The Source-Light-Mass (SLaM) pipelines define a `source_lp[0]` stage
   whose only job is to fit a light-only MGE model to the lens, extras and scaling galaxies. The next stage chains
   from that result to compute luminosities and bound / scale the mass models. See `scripts/group/slam.py`,
   `scripts/group/features/pixelization/slam.py`, and the other group `slam.py` variants for production examples.

This tutorial loads the luminosities from a `scaling_galaxies.csv` written by the simulator (see
`al.galaxy_table_from_csv` further down). In a real analysis the same CSV would be the *output* of one of the patterns
above — `modeling_for_luminosities.py` already writes its result in this format, and the SLAM `source_lp[0]` stage can
similarly emit one.

__Redshifts__

In this example all foreground galaxies are at the same redshift as the lens galaxy, meaning multi-plane lensing is not
used. To enable multi-plane lensing, define per-galaxy redshifts and pass them when constructing each
`af.Model(al.Galaxy, ...)`.

__Group vs Imaging__

This is the **imaging-context** example: there is a single main lens galaxy and all companions live in a single
`extra_galaxies` collection. For the group-scale variant — multiple "main" lens galaxies AND a top-level
`scaling_galaxies` collection separate from `extra_galaxies` — see
`autolens_workspace/scripts/group/features/scaling_relation/modeling.py`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `imaging/start_here.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We use the `dataset/imaging/extra_and_scaling_galaxies` dataset, which contains:

 - a galaxy-scale lens at the origin
 - two close companions (the "individual" tier here)
 - two fainter further-out companions (the "scaling-relation" tier)

The simulator at `scripts/imaging/features/scaling_relation/simulator.py` writes two centre JSON files
(`extra_galaxies_centres.json` and `scaling_galaxies_centres.json`), one per modeling strategy.
"""
dataset_name = "extra_and_scaling_galaxies"
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

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

A 6.0" circular mask, large enough to enclose the lens, the close companions, and the further-out scaling galaxies.
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
__Centres__

The individually-modelled tier loads its centres from a JSON file (a list of (y, x) tuples) — the centres are the only
input the modeling script needs for that tier.
"""
individual_extras_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

print(f"Individually-modelled extras: {individual_extras_centres}")

"""
__Centres + Luminosities (scaling-relation tier)__

The scaling-relation tier needs both centres AND a measured luminosity per galaxy. There are two equally-supported
ways to provide them in PyAutoLens — both shown below so you can pick whichever fits your workflow.

**Option A — CSV via `al.galaxy_table_from_csv` (recommended for non-trivial galaxy counts).** The simulator writes a
`scaling_galaxies.csv` with columns `y, x, luminosity` (and optional `redshift`) alongside the centre JSONs. We load it
in one call which returns a typed `GalaxyTable` with `.centres` (a `Grid2DIrregular`), `.luminosities`, and (optionally)
`.redshifts`. This scales naturally to populations of tens or hundreds of galaxies — the source of truth lives in a
single editable file.

In a real analysis, a prior light-only fit produces this CSV — see
`scripts/group/features/scaling_relation/modeling_for_luminosities.py` for the standalone version of that fit, or the
SLAM `source_lp[0]` stage in `scripts/group/slam.py` for the chained-pipeline equivalent.

**Option B — JSON centres + hardcoded luminosity list (the original API, fine for short, fixed-length tutorials).**
Load the centres from `scaling_galaxies_centres.json` with `al.from_json` (the same loader used for the
individually-modelled tier above) and define the luminosities as a Python list. Concise and obvious for small
populations; awkward once you have more than a handful.

We use Option A by default below. The Option B equivalent is shown commented out — uncomment it (and comment out
Option A) to switch.
"""
# Option A: CSV (recommended)
relational_extras_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
relational_extras_centres = relational_extras_table.centres
relational_extras_luminosity_list = relational_extras_table.luminosities

# Option B: JSON centres + hardcoded luminosities (uncomment to use instead)
# relational_extras_centres = al.from_json(
#     file_path=dataset_path / "scaling_galaxies_centres.json"
# )
# relational_extras_luminosity_list = [0.45, 0.45]
# assert len(relational_extras_luminosity_list) == len(list(relational_extras_centres)), (
#     "Number of luminosities must match number of scaling-relation extra galaxy centres."
# )

print(f"Scaling-relation extras: {relational_extras_centres}")
print(f"Scaling-relation luminosities: {relational_extras_luminosity_list}")

"""
__Lens__

Standard MGE bulge + `Isothermal` mass + `ExternalShear`.
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

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
__Individually-Modelled Extras__

The first tier inside `extra_galaxies`. Each galaxy gets:

 - an MGE bulge with `centre_fixed` (the light is fit but the centre is pinned)
 - an `IsothermalSph` mass with bounded uniform-prior `einstein_radius`

Each adds 1 free Einstein-radius parameter to the model.
"""
extra_galaxies_list = []

for centre in individual_extras_centres:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=tuple(centre)
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = tuple(centre)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.5)

    extra_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

"""
__Scaling-Relation Extras__

The second tier inside `extra_galaxies`, in the reference-anchored Lenstool convention. The relation is defined ONCE
outside the loop, so every galaxy in this tier shares it. Adding more galaxies to this tier does not add free
parameters.

The single free parameter is `einstein_radius_ref` — the Einstein radius of a galaxy *at the reference magnitude*.
Each member's Einstein radius derives from it via its luminosity ratio to a fixed `reference_luminosity` (Lenstool's
`mag0`, an explicit constant — *not* the maximum luminosity of the sample; here a fiducial L* = 1.0), with the
exponent *fixed* at the Faber-Jackson value of 0.5 (einstein_radius ∝ sigma² and sigma ∝ L^(1/4) give
einstein_radius ∝ L^(1/2)). Fixing the exponent avoids the normalization-slope degeneracy; free it as a systematics
test with `scaling_exponent = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)`.

For each galaxy:

 - an MGE bulge with `centre_fixed`
 - an `Isothermal` mass with `einstein_radius = einstein_radius_ref * (luminosity / reference_luminosity) ** 0.5`
"""
einstein_radius_ref = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
scaling_exponent = 0.5
reference_luminosity = 1.0

for relational_centre, relational_luminosity in zip(
    relational_extras_centres, relational_extras_luminosity_list
):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre_fixed=tuple(relational_centre),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = tuple(relational_centre)
    luminosity_ratio = relational_luminosity / reference_luminosity
    mass.einstein_radius = einstein_radius_ref * luminosity_ratio**scaling_exponent

    extra_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

extra_galaxies = af.Collection(extra_galaxies_list)

"""
__Model__

Two top-level components: `galaxies` (lens + source) and `extra_galaxies` (the mixed individual + relational tier).
Keeping all extras in one collection matches the `features/extra_galaxies` naming convention while still letting us
mix the two strategies internally.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The `model.info` attribute prints the composed model. Notice that the first two extras have independent
`einstein_radius` priors, while the last two share `einstein_radius_ref` — the relation in action.
"""
print(model.info)

"""
__Over Sampling__

Adaptive over-sampling at every galaxy centre — lens, individually-modelled extras, and scaling-relation extras alike.
"""
all_centres = (
    [(0.0, 0.0)] + list(individual_extras_centres) + list(relational_extras_centres)
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
    path_prefix=Path("imaging") / "features",
    name="scaling_relation",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Time__

The mixed-strategy model adds a small per-galaxy likelihood overhead but keeps the parameter space compact: only 2
extra parameters from the individually-modelled tier (one Einstein radius each) plus 1 shared parameter from the
scaling-relation tier, no matter how many galaxies sit on it.

GPU log-likelihood evaluation is < 0.005 s per call; CPU is < 0.05 s. Expected end-to-end run time is ~15 minutes on
GPU, ~30 minutes on CPU.

__Model Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

This example showed how to mix two strategies for `extra_galaxies` modeling — individually-modelled and on a shared
scaling relation — within a single `extra_galaxies` collection. The same pattern works with any mass profile and any
measured property (swap the `Isothermal` for `PowerLaw`, or the luminosity for stellar mass, and the structure is
unchanged).

For the production-style luminosity-fitting workflow that produces the `relational_extras_luminosity_list` used here,
see:

 - `autolens_workspace/scripts/group/features/scaling_relation/modeling_for_luminosities.py` — a standalone light-only
   fit that produces per-galaxy total luminosities.
 - `autolens_workspace/scripts/group/slam.py` and `autolens_workspace/scripts/group/features/pixelization/slam.py` —
   the SLAM `source_lp[0]` stage that does the same job inside a chained pipeline.

For the group-scale variant — multiple "main" lens galaxies AND a top-level `scaling_galaxies` collection separate from
`extra_galaxies` — see `autolens_workspace/scripts/group/features/scaling_relation/modeling.py`.
"""
