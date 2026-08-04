"""
Chaining: DSPL (Multi Galaxy)
=============================

This script fits a multi-galaxy double source-plane lens with **two chained searches**, which is how you would
fit one on real data. `modeling.py` in this folder does it in one search, but only by initialising priors at
their true values.

__Why Chain?__

A multi-galaxy DSPL has two hard initialisation problems stacked on top of each other: the mass split between
two co-dominant deflectors, and a second source plane whose image positions depend on the first source's mass.
Asking one search to solve both from broad priors puts it in a parameter space full of local maxima.

Chaining separates them. Search 1 fits only what the bright first ring constrains, using a mask that excludes
the second ring entirely. Search 2 then adds the second source with everything else held fixed, so the only
thing it has to find is the part search 1 could not see.

__Contents__

- **Prerequisites:** What to read first.
- **Dataset:** Load the multi-galaxy DSPL dataset.
- **Centres:** The centres of the co-dominant deflectors.
- **Paths:** Where the two searches write their output.
- **Masking (Search 1):** A mask that excludes the second ring.
- **Model (Search 1):** Deflectors plus the first source only.
- **Search + Analysis + Model-Fit (Search 1):** Run it.
- **Masking (Search 2):** Reload with a mask that includes both rings.
- **Model (Search 2):** Add the second source, fixing what search 1 measured.
- **Search + Analysis + Model-Fit (Search 2):** Run it.
- **Result:** What to check.
- **Wrap Up:** Where to go next.

__Prerequisites__

Read these first:

 - `multi_galaxy/modeling.py` — the multi-galaxy composition every stage here uses.
 - `multi_galaxy/features/advanced/double_source_plane_lens/modeling.py` — what a DSPL model contains, and why
   the second source plane is worth having.
 - `guides/modeling/chaining` — the search-chaining API in general.

__Why The Masking Changes Between Searches__

The mask is the mechanism that makes search 1 easy. With the second ring outside the mask, its pixels contribute
nothing to the likelihood, so `source_1` is not merely un-modelled — it is invisible, and cannot bias the
deflectors by leaving residuals they try to absorb.

This is worth stating because it is easy to get backwards: the small mask is not an approximation to be
tolerated, it is the point of the first search.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `dspl` multi-galaxy dataset.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "dspl"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/multi_galaxy/features/advanced/double_source_plane_lens/simulator.py",
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
__Paths__

Both searches write to the same folder, one sub-folder each.
"""
path_prefix = (
    Path("multi_galaxy") / "features" / "advanced" / "double_source_plane_lens"
)

"""
__Masking (Search 1)__

A tight mask covering both deflectors and the first Einstein ring, but excluding the second ring's arcs.
"""
mask_radius_1 = 1.6

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius_1,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model (Search 1)__

The standard multi-galaxy composition, with `source_0` only. There is no `source_1` in this model at all —
its pixels are outside the mask, so including it would give the search a component with nothing to fit.

`source_0` has light but no mass here. Its mass is constrained by where it puts `source_1`'s images, and those
are not in this mask.
"""
lens_dict_1 = {}

for i, centre in enumerate(main_lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius_1,
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

source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=af.Model(al.lp_linear.SersicCore),
)

model_1 = af.Collection(
    galaxies=af.Collection(**lens_dict_1, shear_galaxy=shear_galaxy, source_0=source_0)
)

print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__source_0",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Masking (Search 2)__

Reload the dataset with the standard 3.0" mask, which contains both rings.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

mask_radius_2 = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius_2,
    )
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 2],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model (Search 2)__

Everything search 1 measured is passed forward as an **instance** — fixed, not refitted. What is added is what
search 1 could not see:

 - `source_0`'s `IsothermalSph` mass, which deflects `source_1`.
 - `source_1` itself, at z=2.0, with a light profile only.

Fixing the deflectors here is a deliberate simplification for a tutorial. In a real analysis you would free them
again in a third search, once `source_1` is in place — which is what `slam.py`'s mass stage does. Leaving them
fixed means this script never gets the benefit of the second ring for the mass split; it only demonstrates the
chaining structure that makes that benefit reachable.
"""
lens_dict_2 = {}

for i, _ in enumerate(main_lens_centres):
    lens_dict_2[f"lens_{i}"] = getattr(result_1.instance.galaxies, f"lens_{i}")

source_0_2 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=result_1.instance.galaxies.source_0.bulge,
    mass=af.Model(al.mp.IsothermalSph),
)

source_0_2.mass.centre = result_1.instance.galaxies.source_0.bulge.centre

source_1 = af.Model(
    al.Galaxy,
    redshift=2.0,
    bulge=af.Model(al.lp_linear.ExponentialCoreSph),
)

model_2 = af.Collection(
    galaxies=af.Collection(
        **lens_dict_2,
        shear_galaxy=result_1.instance.galaxies.shear_galaxy,
        source_0=source_0_2,
        source_1=source_1,
    )
)

print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__source_1",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis_2 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result__

`result_2` holds the three-plane model.

The check specific to this script is whether search 2 actually found the second ring: compare `source_1`'s
fitted centre against where its arcs appear, and look at the residual map outside search 1's mask radius. If
that region still shows arc-shaped residuals, search 2 has not converged onto the second source and nothing
downstream will be meaningful.
"""
print(result_2.info)

aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/advanced/double_source_plane_lens/slam.py` — the full pipeline, which frees the
   deflectors again after the second source is in place.
 - `multi_galaxy/features/advanced/double_source_plane_lens/modeling.py` — the single-search version, and why it
   cheats.
 - `guides/modeling/chaining` — the search-chaining API in general.
 - `imaging/features/advanced/double_source_plane_lens/chaining.py` — the single-deflector version.
"""
