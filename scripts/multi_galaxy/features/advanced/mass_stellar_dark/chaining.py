"""
Chaining: Stellar and Dark Mass (Multi Galaxy)
==============================================

This script fits a multi-galaxy lens with decomposed stellar and dark mass using **two chained searches**: a
total mass model first, the decomposition second.

__Why Chain?__

A decomposition is hard to initialise. The stellar and dark components trade against each other, and with two
co-dominant deflectors there are four components trading rather than two — on top of the mass split, which is
already the regime's standing degeneracy.

A total mass model has none of that. One `Isothermal` per deflector is well constrained by the arcs, converges
easily, and gives an answer that the decomposition must reproduce in sum. Search 2 therefore starts knowing
roughly how much mass each galaxy has, and only has to decide how it is divided between the components.

__Contents__

- **Prerequisites:** What to read first.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Centres:** The centres of the co-dominant deflectors.
- **Paths:** Where the two searches write their output.
- **Model (Search 1):** A total mass model per deflector.
- **Search + Analysis + Model-Fit (Search 1):** Run it.
- **Model (Search 2):** The decomposition, initialised from search 1.
- **Search + Analysis + Model-Fit (Search 2):** Run it.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Prerequisites__

Read these first:

 - `multi_galaxy/modeling.py` — the multi-galaxy composition, including the total mass model search 1 uses.
 - `multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` — what a decomposition contains, and the
   mass-to-light tying choice this script inherits.
 - `guides/modeling/chaining` — the search-chaining API in general.

__What Search 1 Cannot Give Search 2__

The total mass model constrains the *sum* of each deflector's stellar and dark mass. It says nothing about the
split between them, so search 2's component priors cannot be initialised from it — only the light model can,
and through the light the stellar component's shape.

This is the honest limit of chaining here: it removes the initialisation problem for the total, not for the
decomposition. What makes the decomposition itself tractable is tying the mass-to-light ratio across the two
galaxies, which `modeling.py` explains and this script carries through.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `mass_stellar_dark` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector centre. Unlike the DSPL example, both searches use the
same mask — there is no part of the image that only one of them should see.
"""
mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Paths__
"""
path_prefix = Path("multi_galaxy") / "features" / "advanced" / "mass_stellar_dark"

"""
__Model (Search 1)__

The standard multi-galaxy composition from `multi_galaxy/modeling.py`: an MGE per deflector for the light, one
`Isothermal` per deflector for the total mass, the shear in its own galaxy.

Nothing about this search knows a decomposition is coming.
"""
lens_dict_1 = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict_1[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

model_1 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_1,
        shear_galaxy=shear_galaxy,
        source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
    )
)

print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__total_mass",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model (Search 2)__

The decomposition. Each deflector's single `mass` is replaced by an `lmp.Sersic` stellar component and an
`NFWSph` dark halo.

The source is passed through from search 1 as an instance — it is well constrained by then, and refitting it
would let it absorb decomposition errors.

The mass-to-light ratios are tied across the two galaxies, as `modeling.py` argues they should be. The dark
halos are not tied.
"""
lens_dict_2 = {}

for i, centre in enumerate(main_lens_centres):

    bulge = af.Model(al.lmp.Sersic)
    bulge.centre = (centre[0], centre[1])

    dark = af.Model(al.mp.NFWSph)
    dark.centre = (centre[0], centre[1])

    lens_dict_2[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        dark=dark,
    )

for i in range(1, len(lens_dict_2)):
    lens_dict_2[f"lens_{i}"].bulge.mass_to_light_ratio = lens_dict_2[
        "lens_0"
    ].bulge.mass_to_light_ratio

model_2 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_2,
        shear_galaxy=result_1.model.galaxies.shear_galaxy,
        source=result_1.instance.galaxies.source,
    )
)

print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__stellar_dark",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis_2 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result__

The check specific to this script is a comparison between the two searches: each deflector's total mass from
search 2's stellar plus dark components should agree with search 1's `Isothermal`.

If it does not, the decomposition has found a different total rather than a different division of the same
total — which usually means the components have absorbed something else, most often the other galaxy's mass.
"""
print(result_2.info)

aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/mass_stellar_dark/slam.py` — the full pipeline, whose final stage is
   `MASS LIGHT DARK`.
 - `multi_galaxy/features/advanced/mass_stellar_dark/modeling.py` — the single-search version and the tying
   choice.
 - `guides/modeling/chaining` — the search-chaining API in general.
 - `imaging/features/advanced/mass_stellar_dark/chaining.py` — the single-deflector version.
"""
