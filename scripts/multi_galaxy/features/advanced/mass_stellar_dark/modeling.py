"""
Modeling Features: Stellar and Dark Mass (Multi Galaxy)
=======================================================

This script fits a multi-galaxy strong lens where **each co-dominant deflector's mass is decomposed into a
stellar component and a dark matter halo**, rather than described by a single total mass profile.

The stellar component is a light-and-mass profile: one `Sersic` describes both the light and the mass that
traces it, tied together by a `mass_to_light_ratio`. The dark component is an `NFWSph` halo with no light.

__Contents__

- **Advantages:** What a decomposition gives you that a total mass profile does not.
- **Disadvantages:** What it costs.
- **The Choice This Folder Is About:** Whether to tie the mass-to-light ratio across the two deflectors.
- **Model:** Compose the lens model fitted to the data.
- **Dataset, Mask & Over Sampling:** Standard set up.
- **Model Composition:** One `lens_i` per deflector, each with stellar and dark components.
- **Tying The Mass-To-Light Ratio:** How to do it, in three lines.
- **Model Cookbook:** Where the full model-composition API is documented.
- **Search & Analysis:** Configure the fit.
- **Run Time:** Profiling the expected run time of the model-fit.
- **VRAM:** GPU memory used by the model-fit.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Advantages__

A total mass profile measures how much mass is inside the Einstein radius and nothing about what it is made of.
A decomposition measures the stellar and dark components separately, which is what a stellar-mass estimate, an
initial mass function constraint or a halo concentration needs.

It also ties mass to light directly: the stellar component's mass follows its own light profile, so the light
you can see constrains the mass you cannot.

__Disadvantages__

Two components with different radial profiles can trade against each other, because the data constrains their
sum far better than either alone. That is true at galaxy scale and it is why decompositions need good data.

Here it is worse, because there are two galaxies as well as two components — and the multi-galaxy mass split is
already the regime's standing degeneracy.

__The Choice This Folder Is About__

With one lens galaxy, the `mass_to_light_ratio` is one parameter and there is nothing to decide. With two
co-dominant deflectors it is a modelling choice: fit a ratio per galaxy, or tie them to a single shared value.

**Tying is what makes the decomposition tractable here.** The two galaxies' ratios are near-degenerate with each
other — stellar mass can be moved from one galaxy to the other while the total deflection barely changes, which
is the multi-galaxy mass-split degeneracy reappearing inside the stellar component. A shared ratio removes that
direction from the parameter space entirely, rather than leaving the search to find its way along it.

The cost is an assumption: that both galaxies have the same stellar populations. For an interacting pair of
early-types that is defensible. For a pair with visibly different colours it is not, and then you fit the ratios
separately and accept the wider posteriors.

This script fits them **tied**, and shows the untied composition alongside so the switch is one line either way.

__Model__

This script fits an `Imaging` dataset of a 'multi-galaxy' strong lens where:

 - Each co-dominant deflector's light and stellar mass is a linear `lmp.Sersic` with its centre fixed.
 - Each co-dominant deflector's dark matter is an `NFWSph` centred on the same point.
 - The two deflectors share one `mass_to_light_ratio`.
 - The system has a single overall `ExternalShear` at the system centre.
 - The source's light is an MGE.

__Start Here Notebook__

If any code here is unclear, refer to `multi_galaxy/modeling.py` for the multi-galaxy composition and
`imaging/features/advanced/mass_stellar_dark/modeling.py` for the single-deflector decomposition.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `mass_stellar_dark` multi-galaxy dataset — the `simple` pair with each deflector's mass split into a stellar
component and a dark halo.

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

The standard 3.0" mask, over-sampled at every deflector centre.

The mask radius matters more for a decomposition than for a total mass profile. The stellar and dark components
are hardest to tell apart in the centre, where both are steep, and easiest to separate in the outskirts, where
their profiles diverge. Cutting the mask tight removes exactly the pixels that do the separating.
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
__Model Composition__

The standard multi-galaxy composition from `multi_galaxy/modeling.py` — one `lens_i` per deflector in a loop, the
shear in its own `shear_galaxy` — with each deflector's single `mass` replaced by a `bulge` and a `dark`.

`al.lmp.Sersic` is the light-and-mass profile. Its light parameters are the ordinary `Sersic` ones; its
`mass_to_light_ratio` converts that light into a mass distribution with the same shape. `al.lp_linear` has no
equivalent — the mass depends on the absolute intensity, so it cannot be solved linearly.

Both centres are fixed to the deflector's known centre, as elsewhere in this package.
"""
# Main Lens Galaxies:

lens_dict = {}

for i, centre in enumerate(main_lens_centres):

    bulge = af.Model(al.lmp.Sersic)
    bulge.centre = (centre[0], centre[1])

    dark = af.Model(al.mp.NFWSph)
    dark.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        dark=dark,
    )

"""
__Tying The Mass-To-Light Ratio__

The three lines below are the point of this folder.

Assigning one galaxy's `mass_to_light_ratio` to the others makes them the same model parameter, not merely
similarly-primed ones — the search samples a single value used by both galaxies.

To fit them separately, delete this loop. The model then has one more parameter, and that parameter lies along
the near-degenerate direction described at the top of this script, so expect the search to be slower and the
posteriors wider.

The dark halos are **not** tied. A shared stellar population is a defensible assumption for an interacting pair;
a shared halo is not, and tying them would assert the two galaxies have equal dark masses — which is close to
asserting the mass split this whole regime exists to measure.
"""
for i in range(1, len(lens_dict)):
    lens_dict[f"lens_{i}"].bulge.mass_to_light_ratio = lens_dict[
        "lens_0"
    ].bulge.mass_to_light_ratio

# External Shear:

shear_galaxy = af.Model(
    al.Galaxy,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**lens_dict, shear_galaxy=shear_galaxy, source=source)
)

"""
The `info` attribute shows the model in a readable format.

Look for `mass_to_light_ratio` appearing once rather than once per galaxy — that is the tie taking effect.
"""
print(model.info)

"""
__Model Cookbook__

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Search__

The lens model is fitted using the nested sampling algorithm Nautilus.
"""
search = af.Nautilus(
    path_prefix=Path("multi_galaxy") / "features" / "advanced",
    name="mass_stellar_dark",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

"""
__Run Time__

Run times are discussed in full in `multi_galaxy/modeling.py`.

A decomposition evaluates two mass profiles per deflector instead of one, so the deflection half of each
likelihood evaluation roughly doubles. The `NFWSph` is the more expensive of the two.

__VRAM__

The `multi_galaxy/modeling.py` example explains how VRAM is used during GPU-based fitting and how to print the
estimated VRAM required by a model.

The method below prints the VRAM estimate for this analysis and model. It takes 20-30 seconds, so comment it out
once you are familiar with your GPU's limits.
"""
# analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result carries a `bulge` and a `dark` per deflector.

Checks worth making beyond the usual multi-galaxy ones:

 - **The stellar and dark components' relative contributions**, per galaxy, rather than each in isolation. They
   trade against each other, so a tight posterior on one and a loose one on the other usually means the fit has
   pushed the uncertainty into whichever component the data constrains least.
 - **The shared `mass_to_light_ratio` against its prior bounds.** A value pressed against a limit means the tie
   is doing more work than the data supports, and the two galaxies' stellar populations probably do differ.
 - **The dark halos against each other.** They were not tied, so their ratio is a genuine measurement — and it
   is the mass split, restated in terms of the component that carries most of the mass.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/mass_stellar_dark/fit.py` — the decomposed deflection field inspected
   directly, component by component.
 - `multi_galaxy/features/advanced/mass_stellar_dark/chaining.py` — a total mass model first, decomposed second.
 - `multi_galaxy/features/advanced/mass_stellar_dark/slam.py` — the pipeline, whose final stage is
   `MASS LIGHT DARK` rather than `MASS TOTAL`.
 - `imaging/features/advanced/mass_stellar_dark` — the single-deflector decomposition.
"""
