"""
Modeling Features: Linear Light Profiles (Multi Galaxy)
=======================================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted, rather than being sampled by the non-linear search. The solver
always computes the `intensity` values that maximize the likelihood for the other parameters' current values.

This script fits a multi-galaxy strong lens with a linear `Sersic` for each co-dominant deflector and for the
source.

__Contents__

- **Advantages:** What the linear solve removes from the search, and what that is worth here.
- **Disadvantages:** Where the linear solve can mislead you.
- **Model:** Compose the lens model fitted to the data.
- **Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
- **Centres:** Load the deflector centres that drive the model composition loop.
- **Over Sampling:** Adaptive over-sampling at every deflector centre.
- **Model Composition:** Compose the lens model using the Model and Collection API.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **Result:** Overview of the results of the model-fit.
- **Intensities:** Extract the solved-for intensity of each deflector, and the flux ratio between them.
- **Wrap Up:** Summary and where to go next.

__Why This Example Uses Plain Sersics__

`multi_galaxy/modeling.py` already uses linear light profiles — its MGE bases are built from `lp_linear`
Gaussians. So this folder is not introducing the feature to the package; it is showing the feature on its own,
with the simplest possible light model, so you can see what the linear solve does without the MGE composition API
in the way.

A single `Sersic` per galaxy is not the model you should fit to real multi-galaxy data (see
`multi_galaxy/modeling.py`'s `__Improved Lens Model__` section on why symmetric profiles fail, which is worse when
two galaxies' light blends). It is the model that makes this feature legible.

__Advantages__

Each light profile's `intensity` leaves the non-linear parameter space. For the model fitted below:

 - **29** free parameters with standard `lp.Sersic` profiles.
 - **26** with the `lp_linear` equivalents — the two deflectors' intensities and the source's, removed.

Three parameters is a modest saving, and if that were all this did it would barely be worth a folder. The reason it
matters more than the count suggests is *which* parameters they are. `intensity` is strongly degenerate with
`effective_radius` and `sersic_index` within a galaxy: a bigger, shallower profile with lower intensity fits nearly
as well as a smaller, steeper one with higher intensity. The search has to explore that ridge, and it does so
separately for each of the deflectors.

In a multi-galaxy lens those per-galaxy ridges do not stay separate. The two galaxies' light overlaps, so
mis-apportioning flux between them changes the residuals in the region where the arcs are, which the mass model
then absorbs — and the mass split between the deflectors is the least well constrained thing in the fit
(`multi_galaxy/modeling.py`). Removing `intensity` from the search collapses each galaxy's internal degeneracy
before it can couple to the one degeneracy you cannot afford to feed.

__Disadvantages__

The solved intensities are not sampled, so they have no posterior and therefore no errors. If the number you want
*is* a luminosity — or a ratio of luminosities, which for a co-dominant pair it very often is — you get a
maximum-likelihood value with no uncertainty attached to it directly.

The linear solve is also unconstrained in sign by default: it will return whatever intensities maximize the
likelihood, including negative ones. A negative intensity is not physical, and when it appears it is usually
telling you the model is wrong somewhere else. In a multi-galaxy fit the common cause is one deflector's profile
being flexible enough to over-subtract into its neighbour. Check for it; do not ignore it.

__Positive Only Solver__

Many codes which use linear algebra rely on a solver which allows positive and negative values in the solution
(e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means negative surface brightness values can be used to represent a galaxy's light,
which is unphysical.

All linear light profiles use a positive-only solver, so every solved intensity is positive.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light is a linear `Sersic` whose centre is free around its known position
   [6 parameters each].
 - Each deflector's total mass distribution is an `Isothermal` with its centre fixed to that position
   [3 parameters each].
 - The system has a single overall `ExternalShear` at the system centre [2 parameters].
 - The source galaxy's light is a linear `SersicCore` [6 parameters].

__Start Here Notebook__

If any code in this script is unclear, refer to `multi_galaxy/modeling.py` and
`imaging/features/linear_light_profiles/modeling.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the `simple` multi-galaxy dataset — the same pair of co-dominant deflectors fitted by
`multi_galaxy/modeling.py`, so the two fits can be compared directly.
"""
dataset_name = "simple"
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
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
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
__Extra Galaxies Noise Scaling__

The `simple` dataset contains a faint contaminating galaxy whose light is scaled out of the fit before masking,
as explained in full in `multi_galaxy/modeling.py`.

This step is worth doing before a linear fit rather than after. A linear solver hands the model the intensities
that best fit whatever is in the data — including a contaminant's flux, if you leave it there for a nearby
deflector's profile to absorb.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__

Load the centres of the co-dominant deflectors, which drive the model composition loop below.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask__

The standard 3.0" circular mask used throughout the multi-galaxy package, sized by the *combined* Einstein radius
(~1.8") rather than either galaxy's individually.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Over Sampling__

Adaptive over-sampling, centred on **every** deflector — each has a steep central light profile that must be
evaluated accurately, and `centre_list` takes as many centres as you give it.

This matters more for a `Sersic` than for the MGE of `multi_galaxy/modeling.py`. A Sersic with a free
`sersic_index` can become very steeply peaked in its centre, and an under-sampled peak is a systematic the linear
solver will happily absorb into the intensity rather than reject.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Model Composition__

The composition is that of `multi_galaxy/modeling.py` with the MGE bases replaced by single `lp_linear.Sersic`
profiles: one `lens_i` entry per deflector built in a loop over the centres, the shear in its own `shear_galaxy`
at the system centre, and the source separate.

Note that the profiles below take no `intensity` argument. That is the whole API change — the `lp_linear` module
mirrors `lp`, minus `intensity`.

Each bulge centre is free with a `GaussianPrior` around the known position, while each mass centre is fixed to it.
This is the same split `multi_galaxy/modeling.py` uses, and the reasoning is the same: the light is visible and can
locate itself, whereas a free mass centre among several nearby deflectors invites profiles to swap or drift.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = af.Model(al.lp_linear.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.1)

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

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=af.Model(al.lp_linear.SersicCore),
)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
The `info` attribute shows the model in a readable format.

Confirm that no `bulge` has an `intensity` parameter — neither deflector's, nor the source's.
"""
print(model.info)

"""
__Search__

The lens model is fitted using the nested sampling algorithm Nautilus.

`multi_galaxy/modeling.py` uses 200 live points for its MGE model. This model has more free parameters (26 vs 20)
because a free-`sersic_index` Sersic per galaxy is not actually a smaller model than an MGE basis — it is a less
flexible one with a nastier parameter space. We therefore keep 200 rather than reducing it.

That is worth pausing on if you came here expecting linear profiles to make the fit cheaper. They remove three
parameters; the choice of *Sersic over MGE* adds six and makes the remaining ones harder to sample. The two are
independent decisions, and this folder only isolates the first.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features",
    name="linear_light_profiles",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

Linear light profiles are slower per likelihood evaluation than standard profiles, because each evaluation solves
a linear inversion for the intensities. This is offset by the reduced dimensionality of the parameter space, so
the total run time is comparable.

__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, described in `multi_galaxy/modeling.py` and in full in
`autolens_workspace/*/guides/results`.

The `info` below confirms that no `intensity` values were inferred by the non-linear search.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Intensities__

The solved-for intensities are not in the model, so they are read off the fit.

`linear_light_profile_intensity_dict` maps each linear light profile in the model to the intensity the solver
found for it. `tracer_linear_light_profiles_to_light_profiles` gives the same information as an ordinary tracer
whose profiles carry those intensities, which is usually the more convenient form.
"""
fit = result.max_log_likelihood_fit

print(fit.linear_light_profile_intensity_dict)

tracer = fit.tracer_linear_light_profiles_to_light_profiles

"""
__The Flux Ratio__

For a co-dominant pair, the quantity of interest is usually not either intensity but the ratio between the two
galaxies' light. The deflectors are the first `n_main` entries of the tracer's galaxy list, because the model
composed `**lens_dict` first.

Two cautions on the number this prints:

 - It is a **maximum-likelihood** value with no error bar, for the reason given in __Disadvantages__. To put an
   uncertainty on it you need the intensities sampled, or the ratio propagated from a posterior over the shape
   parameters.
 - `intensity` is a surface brightness normalization, not a luminosity. Comparing two galaxies properly means
   integrating each profile, which depends on `effective_radius` and `sersic_index` as well —
   `multi_galaxy/features/scaling_relation/slam.py` shows that integration for an MGE.
"""
n_main = len(list(main_lens_centres))

intensities = [tracer.galaxies[i].bulge.intensity for i in range(n_main)]

for i, intensity in enumerate(intensities):
    print(f"lens_{i} solved intensity = {intensity}")

if intensities[1] != 0.0:
    print(f"\nlens_0 / lens_1 intensity ratio = {intensities[0] / intensities[1]}")

"""
__Negative Intensities__

Check the solve returned physical values. A negative intensity means one profile is subtracting light somewhere,
which in a multi-galaxy fit usually means one deflector's profile has reached across into its neighbour.
"""
if any(intensity < 0.0 for intensity in intensities):
    print(
        "\nWARNING: a deflector's solved intensity is negative. The light model is over-subtracting — "
        "inspect the residuals before trusting any flux measured from this fit."
    )

"""
__Wrap Up__

This script fitted a multi-galaxy lens with linear light profiles. The API change is small — drop `intensity`, use
`lp_linear` — and the parameter saving is modest. What is not modest is which degeneracies it removes: each
galaxy's internal intensity/size ridge, before it can couple to the mass split between the deflectors.

Where to go next:

 - `multi_galaxy/features/linear_light_profiles/fit.py` — the same composition without a search, where the solved
   intensities can be inspected directly against the truth.
 - `multi_galaxy/features/linear_light_profiles/likelihood_function.py` — where in the likelihood the solve
   happens.
 - `multi_galaxy/features/linear_light_profiles/slam.py` — the SLaM pipeline using linear Sersics instead of MGEs.
 - `imaging/features/multi_gaussian_expansion` — the MGE, which is built from linear profiles and is the
   model `multi_galaxy/modeling.py` actually recommends.
 - `imaging/features/linear_light_profiles` — the galaxy-scale walkthrough, with the fuller API tour.
"""
