"""
Chaining: Group Double Einstein Ring
====================================

This script chains two non-linear searches to fit `Imaging` data of a group-scale double Einstein ring lens —
two source galaxies at different redshifts lensed by multiple main lens galaxies at the lens-plane redshift.

In the final model:

 - Each main lens galaxy at z=0.5 has an MGE bulge and an `Isothermal` mass profile, composed via the group
   `lens_dict` convention (one entry per main lens centre loaded from JSON).
 - `source_0` at z=1.0 has an MGE bulge and an `IsothermalSph` mass — it deflects light from `source_1`.
 - `source_1` at z=2.0 has an MGE bulge only.

The two searches break down as follows:

 1) Fit each main lens galaxy's mass and bulge, and `source_0`'s MGE bulge. `source_0`'s mass and `source_1`
    are omitted entirely. A smaller mask removes the second source's emission from the fit.
 2) Pass the search 1 results forward as instances (fixed values), then introduce `source_0`'s mass and
    `source_1`'s MGE bulge as new free parameters. A larger mask includes both source galaxies' emission.

__Why Chain?__

A group-scale double Einstein ring has many more free parameters than either a single-plane group lens OR a
single-lens-galaxy double Einstein ring. A single Nautilus search on the combined model is impractical: the
parameter space is too high-dimensional and local maxima are abundant.

Chaining exploits a key physical observation: ray-tracing of `source_0` is fully independent of `source_1`'s
properties, so we can initialise the lens model + `source_0` first, then introduce `source_1` as a (more
tractable) extension.

__Contents__

- **Dataset & Paths:** Load data; choose the chained-search output path.
- **Main Lens Centres:** Load the two main lens galaxy centres from JSON.
- **Masking (Search 1):** Smaller mask that removes `source_1`.
- **Model (Search 1):** `lens_dict` (MGE bulge + `Isothermal` mass each) plus `source_0` MGE bulge.
- **Search + Result (Search 1):** Run search 1.
- **Masking (Search 2):** Larger mask that includes `source_1`.
- **Model (Search 2):** Search 1 results passed as instances, plus new free `source_0` mass and `source_1` MGE.
- **Search + Result (Search 2):** Run search 2.
- **Wrap Up.**

__Prerequisites__

For background on the canonical group lens chaining workflow and the single-lens double Einstein ring chaining
workflow, see:

 - `autolens_workspace/scripts/group/start_here.py` — group `lens_dict` model composition.
 - `autolens_workspace/scripts/imaging/features/advanced/double_einstein_ring/chaining.py` — single-lens
   double Einstein ring chained search.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the group double Einstein ring `Imaging` dataset.
"""
dataset_name = "double_einstein_ring"
dataset_path = Path("dataset") / "group" / dataset_name

if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/group/features/advanced/double_einstein_ring/simulator.py",
        ],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Main Lens Centres__

Load the two main lens galaxy centres saved by the simulator.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Paths__

All chained-search results are written under this path prefix.
"""
path_prefix = Path("group") / "chaining" / "double_einstein_ring"

"""
__Masking (Search 1)__

A smaller mask that removes the light of `source_1`. The radius is chosen to encompass both main lens galaxies
and the primary Einstein ring around `source_0`, but to exclude the more distant `source_1` arcs.
"""
mask_radius = 2.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model (Search 1)__

Each main lens galaxy gets an MGE bulge (20 Gaussians) and an `Isothermal` mass profile, composed via the
group `lens_dict` convention. `source_0` gets an MGE bulge with 20 Gaussians (no mass yet). `source_1` is
omitted entirely.
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

source_0_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=source_0_bulge)

model_1 = af.Collection(galaxies=af.Collection(**lens_dict_1, source_0=source_0))

print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__lens_dict_source_0_parametric",
    unique_tag=dataset_name,
    n_live=150,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

print(result_1.info)

"""
__Masking (Search 2)__

Reload the dataset with a larger mask that includes `source_1`'s arcs around the lens system.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model (Search 2)__

Each main lens galaxy's mass and bulge are passed forward as `instance` (i.e. fixed at the search 1 results,
not refit). `source_0`'s MGE bulge is also fixed. We then add:

 - `source_0`'s `IsothermalSph` mass (3 free parameters).
 - `source_1`'s MGE bulge centred near (0.0, 0.0), with a narrow Gaussian prior (6 free parameters).
"""
lens_dict_2 = {}

for i, centre in enumerate(main_lens_centres):
    lens_dict_2[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=getattr(result_1.instance.galaxies, f"lens_{i}").bulge,
        mass=getattr(result_1.instance.galaxies, f"lens_{i}").mass,
    )

source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=result_1.instance.galaxies.source_0.bulge,
    mass=al.mp.IsothermalSph,
)

source_1_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=source_1_bulge)

model_2 = af.Collection(
    galaxies=af.Collection(**lens_dict_2, source_0=source_0, source_1=source_1),
)

print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__source_1_parametric",
    unique_tag=dataset_name,
    n_live=200,
)

analysis_2 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

print(result_2.info)

"""
__Wrap Up__

Two chained searches initialised a model for a group-scale double Einstein ring system. The key idea is to
exploit the independence of `source_0`'s ray-tracing from `source_1`'s properties — search 1 nails down the
lens-plane and `source_0` light using a mask that excludes `source_1`, and search 2 fixes that result and adds
the second source plus `source_0`'s mass.

For a fully production-quality fit on real data (including pixelized source reconstructions), see `slam.py` in
the same directory. The single-search `modeling.py` example is for tutorial purposes only — it "cheats" by
initialising priors at the simulator's true values, which is impossible on real data.

__Advanced Chaining__

A more elaborate pipeline (5+ searches, eventually swapping the parametric source models for pixelized
reconstructions) is shown for the single-lens double Einstein ring in
`autolens_workspace/scripts/imaging/features/advanced/double_einstein_ring/chaining.py`. The same logic
transfers to the group-scale case, but no canonical template is currently provided for group DSPL — we are
still working out the most effective way to model these systems.
"""
