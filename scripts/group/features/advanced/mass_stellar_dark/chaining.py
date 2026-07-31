"""
Chaining: Group Mass Stellar Dark
=================================

This script chains two searches to fit `Imaging` data of a 'group-scale' strong lens where each main lens
galaxy is decomposed into stellar + dark matter components.

The two searches break down as follows:

 1) Models each main lens galaxy's light using a `lp.Sersic` bulge (a pure light profile, no stellar mass
    coupling) and the source as an MGE. No dark matter component is included. This fits the geometric and
    photometric properties of each galaxy's bulge with a relatively small parameter space.

 2) Reintroduces the stellar-mass coupling by swapping each galaxy's bulge to a `lmp.Sersic` (a light+mass
    profile), adds an `NFWSph` dark matter halo per galaxy, and adds an `ExternalShear` on `lens_0`. Priors on
    the bulge geometry (centre, ell_comps, intensity, effective_radius, sersic_index) are passed from search
    1, leaving only the `mass_to_light_ratio` and the dark halo parameters as new free parameters in search 2.

__Why Chain?__

A group-scale decomposed-mass model is hard to fit in a single Nautilus run. With two main lens galaxies plus
shear, the search 2 model carries:

 - 2 x 6 = 12 bulge parameters (linear intensity is solved separately).
 - 2 x 2 = 4 dark NFW parameters (`kappa_s`, `scale_radius` per galaxy).
 - 2 x 1 = 2 mass-to-light ratio parameters.
 - 2 external shear parameters.
 - source MGE parameters.

That parameter space has too many degeneracies for a single search starting from broad priors. Chaining lets
search 1 lock down the bulge geometry of every main lens galaxy first, leaving search 2 free to focus on the
mass-to-light ratios, dark NFW parameters, and the shear, with bulge priors tight from search 1.

__Contents__

- **Dataset + Masking:** Load, plot and mask the `Imaging` data.
- **Main Lens Centres:** Load the two main lens galaxy centres from JSON.
- **Paths:** The output path for both chained searches.
- **Model (Search 1):** Per-lens `lp.Sersic` bulge + MGE source. No mass.
- **Search 1:** Nautilus fit, returns `result_1`.
- **Model (Search 2):** Per-lens `lmp.Sersic` bulge (priors from `result_1`) + `NFWSph` dark + shear on
  `lens_0` + MGE source.
- **Search 2:** Nautilus fit, returns `result_2`.
- **Wrap Up:** Summary and pointer to `slam.py`.

__Start Here Notebook__

If any code in this script is unclear, refer to:

 - `autolens_workspace/scripts/group/start_here.py` — the canonical group walkthrough.
 - `autolens_workspace/scripts/imaging/features/advanced/mass_stellar_dark/chaining.py` — the single-galaxy
   decomposed-mass chaining walkthrough this script generalises across multiple main lens galaxies.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_stellar_dark"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/mass_stellar_dark/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
__Main Lens Centres__

Load the two main lens galaxy centres from JSON. These are fixed on each galaxy's bulge centre in both
searches.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

mask_radius = 3.7

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
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Paths__

The path where the results of both chained searches are output:
"""
path_prefix = Path("group") / "chaining" / "mass_stellar_dark"

"""
__Model (Search 1)__

Search 1 fits each main lens galaxy's bulge as a pure `lp.Sersic` light profile (no stellar mass coupling,
no dark NFW). The source is modelled as an MGE. Each bulge's centre is fixed to the main lens centre loaded
from JSON.

For two main lens galaxies, the lens-plane carries 2 x 6 = 12 bulge parameters (linear intensities solved
separately). The source MGE adds its own parameters.
"""
lens_dict_1 = {}

for i, centre in enumerate(main_lens_centres):
    bulge = af.Model(al.lp.Sersic)
    bulge.centre = (centre[0], centre[1])
    lens_dict_1[f"lens_{i}"] = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)

source_bulge_1 = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_1 = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_1)

model_1 = af.Collection(galaxies=af.Collection(**lens_dict_1, source=source_1))

print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__lens_light",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

print(result_1.info)

"""
__Model (Search 2)__

Search 2 reintroduces the stellar-mass coupling. For each main lens galaxy:

 - Replace `lp.Sersic` bulge with `lmp.Sersic` (a light AND mass profile). Pass bulge geometric / photometric
   priors from `result_1.model` via `take_attributes` — `centre`, `ell_comps`, `intensity`,
   `effective_radius`, `sersic_index` all transfer because they share the same names between `lp.Sersic` and
   `lmp.Sersic`. The new parameter introduced by the swap is `mass_to_light_ratio`.
 - Add an `NFWSph` dark matter halo with `centre` fixed to the bulge centre.
 - Add an `ExternalShear` on `lens_0` only.

The source MGE bulge is fixed to its `result_1.instance` value — search 2 does not re-optimise the source
geometry, only the lens-plane mass.
"""
lens_dict_2 = {}

for i, centre in enumerate(main_lens_centres):
    bulge = af.Model(al.lmp.Sersic)
    bulge.take_attributes(source=getattr(result_1.model.galaxies, f"lens_{i}"))
    bulge.centre = (centre[0], centre[1])

    dark = af.Model(al.mp.NFWSph)
    dark.centre = (centre[0], centre[1])

    galaxy_kwargs = dict(redshift=0.5, bulge=bulge, dark=dark)

    if i == 0:
        galaxy_kwargs["shear"] = af.Model(al.mp.ExternalShear)

    lens_dict_2[f"lens_{i}"] = af.Model(al.Galaxy, **galaxy_kwargs)

source_2 = af.Model(
    al.Galaxy, redshift=1.0, bulge=result_1.instance.galaxies.source.bulge
)

model_2 = af.Collection(galaxies=af.Collection(**lens_dict_2, source=source_2))

print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__

You may wish to inspect the `model.info` file of the search 2 model-fit to ensure the priors were passed
correctly.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__mass_stellar_dark",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_2 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

print(result_2.info)

aplt.subplot_tracer(tracer=result_2.max_log_likelihood_tracer, grid=result_2.grids.lp)
aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)
aplt.corner_anesthetic(samples=result_2.samples)

"""
__Wrap Up__

In this example, we passed each main lens galaxy's bulge light model to a per-galaxy decomposed stellar +
dark matter mass model. Search 1 constrained the bulge geometry of every main lens galaxy from the lens
light alone; search 2 then used those tight priors as the starting point for the harder mass + shear fit,
introducing only the `mass_to_light_ratio` and NFW parameters as new free directions.

__SLaM (Source, Light and Mass)__

An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens
modeling processing into a series of fits that first perfect the source model, then the lens light model and
finally the lens mass model. See `slam.py` in this directory.
"""
