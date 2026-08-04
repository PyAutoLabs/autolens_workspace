"""
Modeling Features (Multi Galaxy): Extra Galaxies
================================================

Extends the multi-galaxy lens model with an **extra galaxies tier**: perturbers near the co-dominant pair which
are carried in the model with their own light and mass, their centres fixed to the observed light.

The package's core scripts already handle extra galaxies one way. `multi_galaxy/simulator.py` puts a single
faint contaminant in the `simple` dataset, and `start_here`, `modeling`, `fit` and `likelihood_function` all
demonstrate the `__Extra Galaxies Noise Scaling__` step which scales its light out of the fit. That contaminant
is given **no mass**, deliberately, so the arcs stay clean and every example can treat the field as a pure
two-deflector lens.

This example is the other half. Its dataset gives the extra galaxies **mass as well as light**, and once that is
true, noise scaling stops being sufficient: scaling away every contaminated pixel removes the light, but the
mass is still bending the source. So the question this example answers is not "how do I remove them" — it is
**which tier does this galaxy belong in**, and what happens when you get that wrong.

__The Tier Ladder__

`multi_galaxy/simulator.py` states the judgement: an extra galaxy is a contaminant, a main lens galaxy is a
co-dominant deflector, "and telling them apart is the first judgement you make about a multi-galaxy field — if
in doubt, the test is whether it contributes significantly to the lensing." This package gives you three tiers
to place a galaxy in:

 - **Main lens galaxies** (`galaxies=af.Collection(lens_0=..., lens_1=...)`) — free light and mass, free
   centres. Co-dominant deflectors which set the image configuration.
 - **Extra galaxies** (`extra_galaxies=af.Collection(...)`) — light and mass with **centres fixed** to the
   observed positions and a capped `einstein_radius`. Perturbers on a configuration the main galaxies already
   set. This example.
 - **Scaling galaxies** (`features/scaling_relation`) — a population whose masses follow a luminosity relation
   anchored on the brightest co-dominant deflector, so the tier costs **zero** parameters no matter how many
   galaxies it holds. For many distant, individually-negligible galaxies.

The dataset fitted here is built to sit in the band where the middle tier is the right answer, and the section
**Getting The Tier Wrong** below quantifies what each mistake costs.

__Contents__

- **Dataset:** Load the pair-plus-perturbers dataset (auto-simulating if absent).
- **Mask & Over Sampling:** A mask large enough to admit the extra galaxies, over-sampled at all four centres.
- **Centres:** Load the two centre files — one per tier.
- **Why Not Noise Scaling:** Why the core scripts' lever is not sufficient once an extra galaxy has mass.
- **Model:** The co-dominant pair, via the `lens_{i}` MGE loop of `multi_galaxy/modeling.py`.
- **Extra Galaxies Model:** The perturber tier, with fixed centres and capped Einstein radii.
- **Getting The Tier Wrong:** What promoting or demoting a galaxy actually costs.
- **Search + Analysis / Run Time / Model-Fit / Result.**
- **Wrap Up:** Where to go next up the ladder.

__Model__

This script fits `Imaging` of a multi-galaxy strong lens with a model where:

 - Each of the two co-dominant deflectors has an MGE bulge and an `Isothermal` mass with its centre fixed to the
   observed position.
 - An `ExternalShear` is carried by a single shear galaxy.
 - The source is an MGE.
 - Each extra galaxy has an `ExponentialSph` light profile and an `IsothermalSph` mass, centres fixed
   [2 extra galaxies x (2 light + 1 mass) parameters].

__Start Here Notebook__

If any code in this script is unclear, refer to `multi_galaxy/start_here.ipynb` and `multi_galaxy/modeling.ipynb`
for the co-dominant tier, and `imaging/features/extra_galaxies/modeling.ipynb` for the fuller extra-galaxies API
walkthrough at galaxy scale.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the multi-galaxy dataset `extra_galaxies`: the co-dominant pair, plus two perturbers which have mass.
"""
dataset_name = "extra_galaxies"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/features/extra_galaxies/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres__

Load the centres of both tiers. The split into two files is the tier assignment made concrete:

 - `main_lens_centres.json` — the co-dominant pair. These *initialize* the free centre priors.
 - `extra_galaxies_centres.json` — the perturbers. These *fix* the light and mass profile centres.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(f"Main lens centres:  {main_lens_centres}")
print(f"Extra galaxy centres: {extra_galaxies_centres}")

"""
__Mask & Over Sampling__

A 4.0" mask, larger than the 3.0" used in `multi_galaxy/modeling.py`, so the extra galaxies at ~2.7" from the
field centre fall inside it. Their light has to be in the fitted region for the model to fit it.

The adaptive over sampling scheme is centred on **all four** galaxies. `multi_galaxy/modeling.py` makes the
point that `centre_list` scales with the number of deflectors at no cost; the same applies to the extra
galaxies tier, whose light profiles are just as steeply peaked at their centres.
"""
mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres) + extra_galaxies_centres.in_list,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Why Not Noise Scaling__

The core scripts' lever scales the extra galaxies' pixels so they contribute negligibly to the likelihood. That
is the right tool for a contaminant whose *light* is the only problem, and it is what `multi_galaxy/modeling.py`
demonstrates on the `simple` dataset.

It is not sufficient here, and the size of the shortfall is measurable. Taking this dataset's true tracer and
removing only the extra galaxies' **mass** — keeping their light exactly as simulated — changes the model image
by up to **7.6 sigma** in the worst pixel, with **226 pixels above 3 sigma** and a total sqrt(sum chi^2) of
about **89**. None of that is in the pixels the mask covers. It is in the arcs, where the perturbers' deflection
has moved the source's images.

Noise scaling cannot reach it, because the affected pixels are not the contaminated ones. A model without the
extra-galaxies tier must absorb that signal somewhere, and the only free mass available is the main pair's — so
their Einstein radii and ellipticities distort to compensate. That is the failure mode this tier exists to
prevent, and it is quiet: the fit does not look obviously wrong, it just returns biased deflector masses.

The `mask_extra_galaxies.fits` shipped with this dataset is included so you can run that comparison yourself —
fit with the extra galaxies noise-scaled and omitted from the model, and compare the recovered
`einstein_radius` of `lens_0` and `lens_1` against this example's.

__Model__

The co-dominant tier, exactly as in `multi_galaxy/modeling.py`: one MGE bulge and one `Isothermal` mass per
deflector, composed in a loop over `main_lens_centres`, with a single shear galaxy carrying the external shear
(giving one to each deflector would be redundant and degenerate).
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
__Extra Galaxies Model__

The perturber tier. Each extra galaxy gets an `ExponentialSph` light profile and an `IsothermalSph` mass, both
with their `centre` **fixed** to the loaded position, and the `einstein_radius` capped by a `UniformPrior`.

Both restrictions do real work here, and for reasons sharper than at galaxy scale:

 - **Fixed centres.** A free-centre perturber can wander. At galaxy scale the risk is it drifts onto a source
   image or into the lens's own mass. In a multi-galaxy field there is now a second co-dominant deflector for it
   to drift towards as well, and if it settles near one it becomes degenerate with that galaxy's free mass —
   corrupting exactly the per-deflector measurement the multi-galaxy regime exists to make.
 - **A capped `einstein_radius`.** The upper limit of 0.3" is comfortably above the true values (0.08" and
   0.10") but well below the main pair's (0.9" and 0.8"). Without it, an extra galaxy can climb into
   co-dominant territory and effectively promote itself into the wrong tier mid-fit — the parameterization would
   then no longer mean what the model says it means.

__Untruncated Profiles (Galaxy Scale) vs Truncated dPIE (Group / Cluster Scale)__

The extra galaxies here use the **untruncated** `IsothermalSph` profile — the PyAutoLens-native
parameterization. This is the right choice at galaxy scale: with no group- or cluster-scale host environment,
there is no tidal stripping to encode in the profile. At group and cluster scale the convention flips — member
galaxies are modeled as tidally truncated `dPIEMassSph` profiles (see `group/modeling.py` and
`cluster/modeling.py`), whose `r_cut` truncation encodes the stripping of a member's outer dark matter by the
shared host potential.

The `extra_galaxies` collection is passed alongside `galaxies`. `AnalysisImaging` appends it to the tracer it
builds from each model instance, so the perturbers contribute to the summed deflection field with no further
wiring.
"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Light

    light = af.Model(al.lp.ExponentialSph)
    light.centre = extra_galaxy_centre

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.3)

    extra_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, light=light, mass=mass)
    )

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source),
    extra_galaxies=extra_galaxies,
)

"""
The `info` attribute confirms all three parts: the `lens_0` / `lens_1` co-dominant tier each with free mass, the
shear galaxy, and the `extra_galaxies` tier with fixed centres.
"""
print(model.info)

"""
__Getting The Tier Wrong__

The judgement has a cost in both directions, and neither error announces itself.

**Demoting a co-dominant galaxy** — treating a real second deflector as an extra galaxy, or noise-scaling it
away — caps its Einstein radius and fixes its centre. Its lensing then has to be absorbed by the remaining free
mass, and you recover a single-deflector model of a two-deflector system. This is the error the `multi_galaxy`
package exists to prevent; `multi_galaxy/README.md` describes the systems where it matters.

**Promoting a perturber** — giving a genuine extra galaxy a free centre and uncapped mass — is the subtler
mistake, because the fit will usually still converge and the residuals will look fine. What you lose is
identifiability: a free perturber near a co-dominant deflector is degenerate with that deflector's mass, so the
posterior widens and the per-galaxy Einstein radii start trading against each other. The fit is not wrong so
much as no longer able to answer the question you asked it.

The practical rule follows the test quoted at the top: does it contribute significantly to the *lensing*, not to
the *light*? A bright galaxy with little mass belongs in the extras tier; a faint but massive one may not. If
genuinely unsure, fit it both ways and compare the main deflectors' recovered masses — if they move, the galaxy
was co-dominant.

__Search + Analysis__

The standard setup. The extra galaxies add three free parameters each, so `n_live` is raised relative to
`multi_galaxy/modeling.py`.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="extra_galaxies",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=1000,
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Time__

Adding extra galaxies increases the likelihood evaluation time, because each perturber's light profile image
must be evaluated and blurred and its deflection angles computed. `ExponentialSph` and `IsothermalSph` are both
cheap, so the increase is small.

The larger cost is dimensionality — three free parameters per extra galaxy, which is why the tier does not scale
to large populations. That is precisely the boundary at which `features/scaling_relation` takes over, tying many
galaxies' masses to the brightest deflector's through one shared relation instead.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` confirms both tiers were fitted, with the extra galaxies' Einstein radii inferred alongside the main
pair's.

The number to look at is not the perturbers' own parameters — it is `lens_0` and `lens_1`'s `einstein_radius`.
Those are what the extra-galaxies tier is protecting, and comparing them against a fit which omits the tier is
the direct demonstration of why it is there.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

__Wrap Up__

The extra galaxies API composes cleanly with the multi-galaxy model: the co-dominant tier is untouched, and the
perturbers are added alongside it.

Where to go next on the ladder:

- `autolens_workspace/*/multi_galaxy/features/scaling_relation` — the tier above this one in population size,
  for many distant galaxies on a shared luminosity relation rather than a free parameter each.
- `autolens_workspace/*/imaging/features/extra_galaxies` — the galaxy-scale version, with the fuller API
  walkthrough and both levers side by side including the SLaM pipeline variant.
- `autolens_workspace/*/group` — where the three-tier model stops being an extension and becomes the default,
  with a host halo as an explicit modeling choice.
"""
