"""
Linear Light Profiles: Group SLaM
===================================

This script uses the SLaM pipelines to fit a group-scale strong lens using **linear Sersic light profiles**
instead of Multi-Gaussian Expansion (MGE) profiles for the galaxy light.

The differences from the standard ``group/slam.py`` are:

 - The SOURCE LP PIPELINE 0 uses ``al.lp_linear.Sersic`` for main lens galaxies and
   ``al.lp_linear.SersicSph`` for extra galaxies, instead of MGE models.
 - The LIGHT LP PIPELINE uses ``al.lp_linear.Sersic`` for main lens galaxies and
   ``al.lp_linear.SersicSph`` for extra galaxies, instead of MGE models.

Linear light profiles solve for the ``intensity`` analytically via linear algebra, removing it from the
non-linear parameter space. This provides an alternative to MGE for group-scale modeling: simpler model
composition with fewer basis functions, while still benefiting from the intensity-free parameter space.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**SOURCE LP PIPELINE 0:** Light-only fit using linear Sersic profiles for all galaxies.
**SOURCE LP PIPELINE 1:** Introduces mass and source with light fixed from stage 0.
**SOURCE PIX PIPELINE 1:** Pixelized source fitting (identical to group/slam.py).
**SOURCE PIX PIPELINE 2:** Refined pixelized source (identical to group/slam.py).
**LIGHT LP PIPELINE:** Refits light using linear Sersic profiles.
**MASS TOTAL PIPELINE:** Final mass fit with PowerLaw (identical to group/slam.py).
**Dataset:** Load and plot the strong lens dataset.
**Galaxy Centres:** Load centres from JSON files.
**Mask:** Define the 2D mask applied to the dataset for the model-fit.
**Settings AutoFit:** The settings of autofit.
**SLaM Pipeline:** Run the full pipeline.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (``guides/modeling/slam_start_here``)
  An introduction to the goals, structure, and design philosophy behind SLaM pipelines.

- **Group SLaM** (``group/slam``)
  The standard group-scale SLaM pipeline using MGE profiles.

- **Linear Light Profiles** (``features/linear_light_profiles``)
  How linear light profiles work and their advantages.
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
__SOURCE LP PIPELINE 0__

Light-only fit (no mass, no source) for every galaxy simultaneously, using linear Sersic profiles instead
of MGE. This gives the next search clean fixed light models to build on.

Main lens galaxies use ``al.lp_linear.Sersic`` and extra galaxies use ``al.lp_linear.SersicSph``.
"""


def source_lp_0(
    dataset,
    settings_search,
    main_lens_centres,
    extra_lens_centres,
    scaling_lens_centres,
    mask_radius,
    redshift_lens,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset)

    # --- main lens light models (linear Sersic, light only) ---
    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):
        bulge = af.Model(al.lp_linear.Sersic)
        bulge.centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.1)
        bulge.centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.1)

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None
        )

    # --- extra lens galaxy light models (linear SersicSph) ---
    extra_light_models = []
    for centre in extra_lens_centres:
        bulge = af.Model(al.lp_linear.SersicSph)
        bulge.centre_0 = af.UniformPrior(lower_limit=centre[0] - 0.5, upper_limit=centre[0] + 0.5)
        bulge.centre_1 = af.UniformPrior(lower_limit=centre[1] - 0.5, upper_limit=centre[1] + 0.5)

        extra_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    extra_galaxies = af.Collection(extra_light_models) if extra_light_models else None

    # --- scaling galaxy light models (linear SersicSph) ---
    scaling_light_models = []
    for centre in scaling_lens_centres:
        bulge = af.Model(al.lp_linear.SersicSph)
        bulge.centre_0 = af.UniformPrior(lower_limit=centre[0] - 0.5, upper_limit=centre[0] + 0.5)
        bulge.centre_1 = af.UniformPrior(lower_limit=centre[1] - 0.5, upper_limit=centre[1] + 0.5)

        scaling_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    scaling_galaxies = (
        af.Collection(scaling_light_models) if scaling_light_models else None
    )

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 100 + 30 * len(lens_dict) + 30 * n_extra + 30 * n_scaling
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_lp[0]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        n_like_max=1000000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE LP PIPELINE 1__

Introduces mass and source with light fixed from stage 0. Multiple main-lens galaxies each get an
``Isothermal`` mass; only ``lens_0`` carries an ``ExternalShear``. Extra-galaxy Einstein radii are
bounded by a luminosity-derived prior.

This is identical to the standard group/slam.py ``source_lp_1`` -- the light profiles are fixed from
stage 0 (which used linear Sersic), so no changes are needed here.
"""


def source_lp_1(
    dataset,
    settings_search,
    source_lp_result_0,
    positions,
    pixel_scale,
    redshift_lens,
    redshift_source,
    source_mge_radius,
    n_batch=50,
):
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=0.3)],
    )

    n_main = sum(
        1 for k in vars(source_lp_result_0.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(source_lp_result_0.instance.extra_galaxies))
        if source_lp_result_0.instance.extra_galaxies is not None
        else 0
    )
    n_scaling = (
        len(list(source_lp_result_0.instance.scaling_galaxies))
        if source_lp_result_0.instance.scaling_galaxies is not None
        else 0
    )

    tracer = (
        source_lp_result_0.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    # Source MGE centred on primary lens bulge from stage 0.
    source_bulge = al.model_util.mge_model_from(
        mask_radius=source_mge_radius,
        total_gaussians=30,
        centre=source_lp_result_0.instance.galaxies.lens_0.bulge.centre,
        centre_prior_is_uniform=False,
        centre_sigma=0.6,
    )

    # --- main lens full models (light fixed from stage 0, mass + shear free) ---
    lens_dict = {}
    for i in range(n_main):
        lp0_lens = getattr(source_lp_result_0.instance.galaxies, f"lens_{i}")

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_lens.bulge.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=lp0_lens.bulge,
            disk=lp0_lens.disk,
            point=lp0_lens.point,
            mass=mass,
            shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
        )

    # --- extra lens galaxy models (light fixed, mass bounded by luminosity) ---
    extra_mass_models = []
    for i in range(n_extra):
        lp0_extra = source_lp_result_0.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_extra.bulge.centre

        # For linear Sersic profiles, compute luminosity from the solved profile.
        galaxy_with_intensity = tracer.galaxies[n_main + i]
        total_luminosity = abs(
            galaxy_with_intensity.bulge.luminosity_within_circle_from(radius=10.0)
        ) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_mass_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_extra.bulge, mass=mass
            )
        )

    extra_galaxies = af.Collection(extra_mass_models) if extra_mass_models else None

    # --- scaling lens galaxy models (light fixed, shared luminosity scaling relation) ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_mass_models = []
    for i in range(n_scaling):
        lp0_scaling = source_lp_result_0.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_scaling.bulge.centre

        galaxy_with_intensity = tracer.galaxies[n_main + n_extra + i]
        total_luminosity = abs(
            galaxy_with_intensity.bulge.luminosity_within_circle_from(radius=10.0)
        ) / pixel_scale**2
        mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation

        scaling_mass_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_scaling.bulge, mass=mass
            )
        )

    scaling_galaxies = (
        af.Collection(scaling_mass_models) if scaling_mass_models else None
    )
    source = af.Model(
        al.Galaxy, redshift=redshift_source, bulge=source_bulge
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
    )

    n_extra_model = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling_model = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 150 + 30 * n_main + 30 * n_extra_model + 30 * n_scaling_model

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
        n_like_max=200000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to ``group/slam.py``. The pixelization pipelines fix the light from previous stages and
do not change light profile type.
"""


def source_pix_1(
    dataset,
    mask,
    settings_search,
    source_lp_result_1,
    over_sample_size,
    pixel_scale,
    mask_radius,
    positions,
    n_batch=20,
):
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30
    signal_to_noise_threshold = 3.0

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
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
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result_1.positions_likelihood_from(
                factor=2.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    n_lenses = sum(
        1 for k in vars(source_lp_result_1.instance.galaxies) if k.startswith("lens_")
    )

    lens_dict = {}
    for i in range(n_lenses):
        lp_lens_instance = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")
        lp_lens_model = getattr(source_lp_result_1.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=lp_lens_model.mass,
            mass_result=lp_lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_lens_instance.redshift,
            bulge=lp_lens_instance.bulge,
            disk=lp_lens_instance.disk,
            point=lp_lens_instance.point,
            mass=mass,
            shear=lp_lens_model.shear,
        )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
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
        extra_galaxies=source_lp_result_1.model.extra_galaxies,
        scaling_galaxies=source_lp_result_1.model.scaling_galaxies,
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

Identical to ``group/slam.py``. Adapt data for the Hilbert image mesh is capped at a S/N threshold
of 3.0 to prevent over-concentration of source pixels.
"""


def source_pix_2(
    dataset,
    mask,
    settings_search,
    source_lp_result_1,
    source_pix_result_1,
    over_sample_size,
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
        over_sample_size_lp=over_sample_size,
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
        lp_lens_instance = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")
        pix1_lens_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_lens_instance.redshift,
            bulge=lp_lens_instance.bulge,
            disk=lp_lens_instance.disk,
            point=lp_lens_instance.point,
            mass=pix1_lens_instance.mass,
            shear=pix1_lens_instance.shear,
        )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
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
        scaling_galaxies=source_pix_result_1.instance.scaling_galaxies,
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
__LIGHT LP PIPELINE__

Refits the light using linear Sersic profiles instead of MGE. Main lens galaxies get a fresh
``al.lp_linear.Sersic`` and extra galaxies get a fresh ``al.lp_linear.SersicSph``, with mass
fixed from ``source_pix[1]``.
"""


def light_lp(
    dataset,
    settings_search,
    source_lp_result_0,
    source_pix_result_1,
    source_pix_result_2,
    adapt_images,
    mask_radius,
    redshift_lens,
    n_batch=20,
):
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
    )

    n_lenses = sum(
        1 for k in vars(source_pix_result_1.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(source_pix_result_1.instance.extra_galaxies))
        if source_pix_result_1.instance.extra_galaxies is not None
        else 0
    )

    # --- main lens light models (fresh linear Sersic) ---
    lens_bulge_list = []
    for i in range(n_lenses):
        bulge = af.Model(al.lp_linear.Sersic)
        lens_bulge_list.append(bulge)

    # --- extra lens galaxy light models (fresh linear SersicSph, mass fixed) ---
    extra_light_models = []
    for i in range(n_extra):
        pix1_extra = source_pix_result_1.instance.extra_galaxies[i]
        bulge = af.Model(al.lp_linear.SersicSph)
        bulge.centre = pix1_extra.mass.centre

        extra_light_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=pix1_extra.mass
            )
        )

    extra_galaxies = af.Collection(extra_light_models) if extra_light_models else None

    source = al.util.chaining.source_custom_model_from(
        result=source_pix_result_2, source_is_model=False
    )

    lens_dict = {}
    for i in range(n_lenses):
        lens_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_bulge_list[i],
            disk=None,
            point=None,
            mass=lens_instance.mass,
            shear=lens_instance.shear,
        )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=source_pix_result_2.instance.scaling_galaxies,
    )

    n_live = 300 + 100 * (n_lenses - 1)

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=n_live,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to ``group/slam.py``. Extra galaxies receive a new luminosity-bounded ``Isothermal`` mass
(using ``light[1]`` luminosities) and scaling galaxies receive a new shared luminosity scaling relation.
"""


def mass_total(
    dataset,
    settings_search,
    source_pix_result_1,
    source_pix_result_2,
    light_result,
    adapt_images,
    positions,
    pixel_scale,
    redshift_lens,
    n_batch=20,
):
    n_lenses = sum(
        1 for k in vars(light_result.instance.galaxies) if k.startswith("lens_")
    )
    n_extra = (
        len(list(light_result.instance.extra_galaxies))
        if light_result.instance.extra_galaxies is not None
        else 0
    )
    n_scaling = (
        len(list(light_result.instance.scaling_galaxies))
        if light_result.instance.scaling_galaxies is not None
        else 0
    )

    # --- extra galaxies: fixed light, free mass ---
    # For linear Sersic profiles, use the tracer with solved intensities.
    tracer = (
        light_result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    extra_mass_models = []
    for i in range(n_extra):
        light_extra = light_result.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_extra.bulge.centre

        galaxy_with_intensity = tracer.galaxies[n_lenses + 1 + i]
        total_luminosity = abs(
            galaxy_with_intensity.bulge.luminosity_within_circle_from(radius=10.0)
        ) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_mass_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=light_extra.bulge, mass=mass
            )
        )

    extra_galaxies = af.Collection(extra_mass_models) if extra_mass_models else None

    # --- scaling galaxies: fixed light, free shared scaling relation ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_mass_models = []
    for i in range(n_scaling):
        light_scaling = light_result.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_scaling.bulge.centre

        galaxy_with_intensity = tracer.galaxies[n_lenses + 1 + n_extra + i]
        total_luminosity = abs(
            galaxy_with_intensity.bulge.luminosity_within_circle_from(radius=10.0)
        ) / pixel_scale**2
        mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation

        scaling_mass_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=light_scaling.bulge, mass=mass
            )
        )

    scaling_galaxies = (
        af.Collection(scaling_mass_models) if scaling_mass_models else None
    )

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            light_result.positions_likelihood_from(
                factor=3.0, positions=positions, minimum_threshold=0.2
            )
        ],
    )

    source = al.util.chaining.source_from(result=source_pix_result_2)

    lens_dict = {}
    for i in range(n_lenses):
        lens_model = getattr(source_pix_result_1.model.galaxies, f"lens_{i}")
        light_lens_instance = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_lens_instance.bulge,
            disk=light_lens_instance.disk,
            point=light_lens_instance.point,
            mass=mass,
            shear=lens_model.shear,
        )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict, source=source),
        extra_galaxies=extra_galaxies,
        scaling_galaxies=scaling_galaxies,
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

Load, plot and mask the ``Imaging`` data.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

pixel_scale = 0.1
mask_radius = 6.0
mask_centre = (0.0, 0.0)
redshift_lens = 0.5
redshift_source = 1.0
source_mge_radius = 1.0
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

main_lens_centres.json         -- required; determines the number of main lenses.
extra_galaxies_centres.json   -- optional; empty list if absent.
scaling_galaxies_centres.json -- optional; empty list if absent.

All three files contain a list of [y, x] arcsecond coordinates.
"""
main_lens_centres = _load_centres(dataset_path / "main_lens_centres.json")
extra_lens_centres = _load_centres(dataset_path / "extra_galaxies_centres.json")
scaling_lens_centres = _load_centres(dataset_path / "scaling_galaxies_centres.json")

all_galaxy_centres = al.Grid2DIrregular(
    main_lens_centres.in_list
    + extra_lens_centres.in_list
    + scaling_lens_centres.in_list
)

positions = al.Grid2DIrregular(al.from_json(file_path=dataset_path / "positions.json"))

"""
__Mask__
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
    centre=mask_centre,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.1, 0.3],
    centre_list=list(all_galaxy_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

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
"""
source_lp_result_0 = source_lp_0(
    dataset=dataset,
    settings_search=settings_search,
    main_lens_centres=main_lens_centres,
    extra_lens_centres=extra_lens_centres,
    scaling_lens_centres=scaling_lens_centres,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
)

source_lp_result_1 = source_lp_1(
    dataset=dataset,
    settings_search=settings_search,
    source_lp_result_0=source_lp_result_0,
    positions=positions,
    pixel_scale=pixel_scale,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
    source_mge_radius=source_mge_radius,
)

source_pix_result_1, dataset, adapt_images = source_pix_1(
    dataset=dataset,
    mask=mask,
    settings_search=settings_search,
    source_lp_result_1=source_lp_result_1,
    over_sample_size=over_sample_size,
    pixel_scale=pixel_scale,
    mask_radius=mask_radius,
    positions=positions,
    n_batch=n_batch,
)

source_pix_result_2, dataset, adapt_images = source_pix_2(
    dataset=dataset,
    mask=mask,
    settings_search=settings_search,
    source_lp_result_1=source_lp_result_1,
    source_pix_result_1=source_pix_result_1,
    over_sample_size=over_sample_size,
    pixel_scale=pixel_scale,
    mask_radius=mask_radius,
    n_batch=n_batch,
)

light_result = light_lp(
    dataset=dataset,
    settings_search=settings_search,
    source_lp_result_0=source_lp_result_0,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    adapt_images=adapt_images,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
    n_batch=n_batch,
)

mass_result = mass_total(
    dataset=dataset,
    settings_search=settings_search,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    light_result=light_result,
    adapt_images=adapt_images,
    positions=positions,
    pixel_scale=pixel_scale,
    redshift_lens=redshift_lens,
    n_batch=n_batch,
)

"""
Finish.
"""
