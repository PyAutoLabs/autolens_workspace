"""
No Lens Light: Group SLaM
=========================

This script uses the SLaM pipelines to fit a group-scale strong lens where none of the lens galaxies have
visible light emission. In the group context, "no lens light" means that **all** main lens galaxies **and**
all extra galaxies are modeled with mass profiles only.

This is the group-scale analogue of `imaging/features/no_lens_light/slam.py`. The pipeline is substantially
simplified compared to the standard group SLaM (`group/slam.py`) because:

 - There is no `source_lp_0` stage (no light-only fit needed since there is no lens light).
 - There is no `light_lp` stage (no lens light to refit).
 - The pipeline goes directly: source_lp -> source_pix_1 -> source_pix_2 -> mass_total.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**This Script:** Using a SOURCE LP PIPELINE (one search), SOURCE PIX PIPELINE (two searches), and.
**SOURCE LP PIPELINE:** Fits mass + source directly. No lens light for any galaxy.
**SOURCE PIX PIPELINE 1:** Pixelized source, mass carried forward. No lens light.
**SOURCE PIX PIPELINE 2:** Refined pixelized source. No lens light.
**MASS TOTAL PIPELINE:** Final mass fit with PowerLaw. No lens light.
**Dataset:** Load and plot the strong lens dataset.
**Galaxy Centres:** Load centres from JSON files.
**Mask:** Define the 2D mask applied to the dataset for the model-fit.
**Settings AutoFit:** The settings of autofit, which controls the output paths, parallelization, database use, etc.
**SLaM Pipeline:** Overview of slam pipeline for this example.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines
  and how they integrate into strong-lens modeling.

- **Group** (`group/modeling`):
    How we model group-scale strong lenses in PyAutoLens, including how we include extra galaxies in
    the lens model.

- **No Lens Light** (`imaging/features/no_lens_light`):
    How the SLaM pipeline is adapted when no lens light is present.

__This Script__

Using a SOURCE LP PIPELINE (one search), SOURCE PIX PIPELINE (two searches) and TOTAL MASS PIPELINE this
SLaM modeling script fits `Imaging` data of a group-scale strong lens where in the final model:

 - Each main lens galaxy has a `PowerLaw` total mass — no light.
 - Each extra galaxy has a bounded `IsothermalSph` mass — no light.
 - The source galaxy's light is a Delaunay `Pixelization` with `AdaptSplit` regularization.

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def _load_centres(path):
    """Load a centres JSON file, returning an empty list if the file is absent."""
    try:
        return al.Grid2DIrregular(al.from_json(file_path=path))
    except FileNotFoundError:
        return al.Grid2DIrregular([])


"""
__SOURCE LP PIPELINE__

Fits mass + source directly. Because no galaxy has light, there is no need for a light-only stage
(`source_lp_0` in the standard group SLaM). We go straight to fitting mass and source simultaneously.

Multiple main-lens galaxies each get an `Isothermal` mass; only `lens_0` carries an `ExternalShear`.
Extra-galaxy Einstein radii are bounded by a uniform prior.
"""


def source_lp(
    dataset,
    settings_search,
    main_lens_centres,
    extra_lens_centres,
    mask_radius,
    redshift_lens,
    redshift_source,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset)

    # Source MGE
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )

    # --- main lens mass models (no light) ---
    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):

        mass = af.Model(al.mp.Isothermal)

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            mass=mass,
            shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
        )

    # --- extra galaxy mass models (no light) ---
    extra_mass_models = []
    for centre in extra_lens_centres:

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

        extra_mass_models.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

    extra_galaxies = af.Collection(extra_mass_models) if extra_mass_models else None
    source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=extra_galaxies,
    )

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_live = 150 + 30 * len(lens_dict) + 30 * n_extra

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Pixelized source with mass carried forward from `source_lp`. No lens light to fix — the galaxies
remain mass-only. Extra galaxy models are carried forward as free `model` parameters.
"""


def source_pix_1(
    dataset,
    mask,
    settings_search,
    source_lp_result,
    pixel_scale,
    mask_radius,
    positions,
    n_batch=20,
):
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30
    signal_to_noise_threshold = 3.0

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=2.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    n_lenses = sum(
        1 for k in vars(source_lp_result.instance.galaxies) if k.startswith("lens_")
    )

    lens_dict = {}
    for i in range(n_lenses):
        lp_lens_model = getattr(source_lp_result.model.galaxies, f"lens_{i}")
        lp_lens_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=lp_lens_model.mass,
            mass_result=lp_lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_lens_instance.redshift,
            mass=mass,
            shear=lp_lens_model.shear,
        )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(
                pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total
            ),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=source_lp_result.model.extra_galaxies,
    )

    n_live = 150 + 50 * (n_lenses - 1)

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
    return result, dataset, adapt_images


"""
__SOURCE PIX PIPELINE 2__

Refined pixelized source. The adapt data for the Hilbert image mesh is capped at a S/N threshold of 3.0
to prevent over-concentration of source pixels. Extra galaxy models are fixed as instances from `source_pix[1]`.
"""


def source_pix_2(
    dataset,
    mask,
    settings_search,
    source_lp_result,
    source_pix_result_1,
    pixel_scale,
    mask_radius,
    n_batch=20,
):
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30
    signal_to_noise_threshold = 3.0
    signal_to_noise_threshold_image_mesh = 3.0

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_data_snr_max = galaxy_image_name_dict["('galaxies', 'source')"]
    adapt_data_snr_max[adapt_data_snr_max > signal_to_noise_threshold_image_mesh] = (
        signal_to_noise_threshold_image_mesh
    )

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=adapt_data_snr_max
    )

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    n_lenses = sum(
        1 for k in vars(source_pix_result_1.instance.galaxies) if k.startswith("lens_")
    )

    lens_dict = {}
    for i in range(n_lenses):
        pix1_lens_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=pix1_lens_instance.redshift,
            mass=pix1_lens_instance.mass,
            shear=pix1_lens_instance.shear,
        )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(
                pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total
            ),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)
    return result, dataset, adapt_images


"""
__MASS TOTAL PIPELINE__

Final mass fit with PowerLaw profiles for the main lens galaxies. No lens light is included.
Extra galaxies receive new bounded Einstein radii.

Note: there is **no** `light_lp` stage in this pipeline because there is no lens light to refit.
"""


def mass_total(
    dataset,
    settings_search,
    source_pix_result_1,
    source_pix_result_2,
    adapt_images,
    positions,
    redshift_lens,
    n_batch=20,
):
    n_lenses = sum(
        1 for k in vars(source_pix_result_1.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(source_pix_result_1.instance.extra_galaxies))
        if source_pix_result_1.instance.extra_galaxies is not None
        else 0
    )

    # --- extra galaxies: mass only, new bounded Einstein radii ---
    extra_mass_models = []
    for i in range(n_extra):
        pix1_extra = source_pix_result_1.instance.extra_galaxies[i]

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = pix1_extra.mass.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

        extra_mass_models.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

    extra_galaxies = af.Collection(extra_mass_models) if extra_mass_models else None

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_1.positions_likelihood_from(
                factor=3.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    lens_dict = {}
    for i in range(n_lenses):
        lens_model = getattr(source_pix_result_1.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            mass=mass,
            shear=lens_model.shear,
        )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=extra_galaxies,
    )

    n_live = 200 + 100 * (n_lenses - 1)

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/features/no_lens_light/simulator.py"],
        check=True,
    )

pixel_scale = 0.1
mask_radius = 7.5
redshift_lens = 0.5
redshift_source = 1.0
n_batch = 20

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

main_lens_centres.json         — required; determines the number of main lenses.
extra_galaxies_centres.json   — optional; empty list if absent.

All files contain a list of [y, x] arcsecond coordinates.
"""
main_lens_centres = _load_centres(dataset_path / "main_lens_centres.json")
extra_lens_centres = _load_centres(dataset_path / "extra_galaxies_centres.json")

positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

"""
__Mask__

Define the 2D mask applied to the dataset for the model-fit. We use a 7.5 arcsecond circular mask.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("group") / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__SLaM Pipeline__

The code below calls the full SLaM PIPELINE for the no-lens-light group case.

The pipeline is:

 1. `source_lp`: Fit mass + MGE source directly — no lens light for any galaxy.
 2. `source_pix_1`: Pixelized source, mass carried forward.
 3. `source_pix_2`: Refined pixelized source.
 4. `mass_total`: Final PowerLaw mass fit.

There is **no** `source_lp_0` (light-only) stage and **no** `light_lp` stage because there is no lens light.
"""
source_lp_result = source_lp(
    dataset=dataset,
    settings_search=settings_search,
    main_lens_centres=main_lens_centres,
    extra_lens_centres=extra_lens_centres,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1, dataset, adapt_images = source_pix_1(
    dataset=dataset,
    mask=mask,
    settings_search=settings_search,
    source_lp_result=source_lp_result,
    pixel_scale=pixel_scale,
    mask_radius=mask_radius,
    positions=positions,
    n_batch=n_batch,
)

source_pix_result_2, dataset, adapt_images = source_pix_2(
    dataset=dataset,
    mask=mask,
    settings_search=settings_search,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    pixel_scale=pixel_scale,
    mask_radius=mask_radius,
    n_batch=n_batch,
)

mass_result = mass_total(
    dataset=dataset,
    settings_search=settings_search,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    adapt_images=adapt_images,
    positions=positions,
    redshift_lens=redshift_lens,
    n_batch=n_batch,
)

"""
Finish.
"""
