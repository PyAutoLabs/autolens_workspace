"""
SLaM (Source, Light and Mass): Group Scale
==========================================

This script uses the SLaM pipelines to fit a group-scale strong lens, including extra galaxies and
scaling galaxies surrounding the main lens whose light and mass are both modeled.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with.
**Extra Galaxies and Scaling Galaxies:** This group-scale SLaM pipeline handles two distinct categories of companion galaxy, which differ in.
**This Script:** Using a SOURCE LP PIPELINE (two searches), SOURCE PIX PIPELINE (two searches), LIGHT LP PIPELINE.
**SOURCE LP PIPELINE 0:** Not present in `slam_start_here.py`.
**SOURCE LP PIPELINE 1:** Equivalent to `source_lp` in `slam_start_here.py`, except lens light is fixed from `source_lp[0]`.
**SOURCE PIX PIPELINE 1:** Equivalent to `source_pix_1` in `slam_start_here.py`, except a Hilbert image mesh is used instead.
**SOURCE PIX PIPELINE 2:** Identical to `source_pix_1` above, except the adapt data for the Hilbert image mesh is capped at a.
**LIGHT LP PIPELINE:** Identical to `light_lp` in `slam_start_here.py`, except extra galaxies receive a fresh free MGE.
**MASS TOTAL PIPELINE:** Identical to `mass_total` in `slam_start_here.py`, except extra galaxies receive a new.
**Dataset:** Load and plot the strong lens dataset.
**Galaxy Centres:** main_lens_centres.json — required; determines the number of main lenses.
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

__Extra Galaxies and Scaling Galaxies__

This group-scale SLaM pipeline handles two distinct categories of companion galaxy, which differ in
how their masses are parameterized:

**Extra Galaxies**

Extra galaxies are a small number of nearby companions whose light and mass are modeled individually.
In each pipeline stage, they receive a free MGE light profile and an `Isothermal` mass profile whose
Einstein radius prior is bounded by a value derived from the galaxy's luminosity:

    upper_limit = min(5 * 0.5 * total_luminosity^0.6, 5.0)

This luminosity-informed bound prevents unphysically large mass assignments while keeping the mass
free per galaxy. The extra-galaxy models are stored in `model.extra_galaxies` (a `Collection`).

**Scaling Galaxies**

Scaling galaxies are a larger ensemble of companions whose masses are constrained through a shared
luminosity-to-mass scaling relation rather than being individually free. They each carry a free MGE
light profile, but their Einstein radii follow:

    einstein_radius = scaling_factor * total_luminosity^scaling_relation

where `scaling_factor` and `scaling_relation` are two shared free parameters whose priors are
`UniformPrior(0, 0.5)` and `UniformPrior(0, 2)` respectively. This reduces the number of mass
parameters considerably when many companion galaxies are present. Scaling-galaxy models are stored
in `model.scaling_galaxies`.

The choice between these two categories is determined by which JSON file each galaxy's centre
appears in (`extra_galaxies_centres.json` vs `scaling_galaxies_centres.json`).

**Comparison with the Galaxy-Scale SLaM Pipeline**

The galaxy-scale extra-galaxies SLaM pipeline (`features/extra_galaxies/slam`) models each
companion galaxy with a fully free `IsothermalSph` mass — it does not use luminosity bounds or a
shared scaling relation. The group-scale pipeline introduced here is appropriate when:

- there are more companion galaxies than can be modeled independently, or
- prior physical knowledge motivates a luminosity-to-mass scaling.

__This Script__

Using a SOURCE LP PIPELINE (two searches), SOURCE PIX PIPELINE (two searches), LIGHT LP PIPELINE
and TOTAL MASS PIPELINE this SLaM modeling script fits `Imaging` data of a group-scale strong lens
where in the final model:

 - Each main lens galaxy has a free MGE bulge and a `PowerLaw` total mass.
 - Each extra galaxy has a free MGE bulge and a luminosity-bounded `Isothermal` mass.
 - Each scaling galaxy has a free MGE bulge and a mass set by a shared scaling relation.
 - The source galaxy's light is a Delaunay `Pixelization` with `AdaptSplit` regularization.

__Start Here Notebook__

If any code in this script is unclear, refer to the `guides/modeling/slam_start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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

Not present in `slam_start_here.py`. Fits light only — no mass, no source — for every galaxy
simultaneously, giving the next search clean fixed light models to build on.

Fits multiple main-lens galaxies (`lens_0`, `lens_1`, ...) under `galaxies`, extra galaxies
under `extra_galaxies`, and scaling galaxies under `scaling_galaxies`. `n_live` scales with
the total number of galaxies across all three categories.
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

    # --- main lens light models (one per centre, light only) ---
    lens_light_models = []
    for centre in main_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=False,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )
        lens_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None)
        )

    # --- extra lens galaxy light models ---
    extra_light_models = []
    for centre in extra_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            ell_comps_prior_is_uniform=True,
        )
        extra_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    extra_galaxies = af.Collection(extra_light_models) if extra_light_models else None

    # --- scaling galaxy light models ---
    scaling_light_models = []
    for centre in scaling_lens_centres:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            ell_comps_prior_is_uniform=True,
        )
        scaling_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    scaling_galaxies = (
        af.Collection(scaling_light_models) if scaling_light_models else None
    )

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_scaling = len(scaling_galaxies) if scaling_galaxies is not None else 0
    n_live = 100 + 30 * len(lens_light_models) + 30 * n_extra + 30 * n_scaling

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_light_models)}
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

Equivalent to `source_lp` in `slam_start_here.py`, except lens light is fixed from
`source_lp[0]` rather than free, and mass and source are introduced here for the first time.

Multiple main-lens galaxies each get an `Isothermal` mass; only `lens_0` carries an
`ExternalShear`. Extra-galaxy Einstein radii are bounded by a luminosity-derived prior
(`min(5 * 0.5 * L^0.6, 5.0)`). Scaling galaxies share two free parameters,
`scaling_factor` and `scaling_relation`, so their masses follow
`einstein_radius = scaling_factor * luminosity^scaling_relation`.
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
    # Only lens_0 carries the ExternalShear; one shear per group system.
    lens_full_models = []
    for i in range(n_main):
        lp0_lens = getattr(source_lp_result_0.instance.galaxies, f"lens_{i}")

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_lens.bulge.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)

        lens_full_models.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lp0_lens.bulge,
                disk=lp0_lens.disk,
                point=lp0_lens.point,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
            )
        )

    # --- extra lens galaxy models (light fixed, mass bounded by luminosity) ---
    # Tracer order: [lens_0..lens_{n_main-1}, extra_0..extra_{n_extra-1}, scaling_0..]
    extra_mass_models = []
    for i in range(n_extra):
        lp0_extra = source_lp_result_0.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lp0_extra.bulge.centre
        mass.ell_comps = lp0_extra.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in tracer.galaxies[n_main + i].bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_mass_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=lp0_extra.bulge, mass=mass)
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
        mass.ell_comps = lp0_scaling.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in tracer.galaxies[n_main + n_extra + i].bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = scaling_factor * total_luminosity**scaling_relation

        scaling_mass_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_scaling.bulge, mass=mass
            )
        )

    scaling_galaxies = (
        af.Collection(scaling_mass_models) if scaling_mass_models else None
    )

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_full_models)}
    lens_dict["source"] = af.Model(
        al.Galaxy, redshift=redshift_source, bulge=source_bulge
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
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

Equivalent to `source_pix_1` in `slam_start_here.py`, except a Hilbert image mesh is used
instead of `RectangularAdaptDensity`, with edge-point padding of 30 pixels. The Hilbert pixel
count is set by `al.model_util.hilbert_pixels_from_pixel_scale`.

Pixelization over-sampling is signal-adaptive: pixels above the S/N threshold use sub-size 4,
the rest sub-size 2. The re-sampled dataset and adapt_images are returned alongside the result.
Extra and scaling galaxy models are carried forward as free `model` parameters, not fixed instances.
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

    lens_dict["source"] = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
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

Identical to `source_pix_1` above, except the adapt data for the Hilbert image mesh is capped
at a S/N threshold of 3.0 to prevent over-concentration of source pixels on the brightest peak,
and extra and scaling galaxy models are fixed as instances from `source_pix[1]`.
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

    lens_dict["source"] = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=af.Model(
            al.Pixelization,
            mesh=al.mesh.Delaunay(pixels=hilbert_pixels, zeroed_pixels=edge_pixels_total),
            regularization=af.Model(al.reg.AdaptSplit),
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
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

Identical to `light_lp` in `slam_start_here.py`, except extra galaxies receive a fresh free
MGE bulge (centred on the `source_pix[1]` mass centre) with mass fixed from `source_pix[1]`,
and scaling galaxies are fully fixed from `source_pix[2]`.
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

    # --- main lens light models (MGE centred on stage-0 bulge centre) ---
    lens_bulge_list = []
    for i in range(n_lenses):
        lp0_lens = getattr(source_lp_result_0.instance.galaxies, f"lens_{i}")
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=lp0_lens.bulge.centre,
        )
        lens_bulge_list.append(bulge)

    # --- extra lens galaxy light models (free MGE, mass fixed from source_pix[1]) ---
    extra_light_models = []
    for i in range(n_extra):
        pix1_extra = source_pix_result_1.instance.extra_galaxies[i]
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=True,
            centre=pix1_extra.mass.centre,
        )
        extra_light_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=pix1_extra.mass)
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

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
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

Identical to `mass_total` in `slam_start_here.py`, except extra galaxies receive a new
luminosity-bounded `Isothermal` mass (using `light[1]` luminosities) and scaling galaxies
receive a new shared luminosity scaling relation, both paired with their fixed `light[1]` bulge.
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
    # Total mass model for each main lens galaxy.
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
    extra_mass_models = []
    for i in range(n_extra):
        light_extra = light_result.instance.extra_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_extra.bulge.centre
        mass.ell_comps = light_extra.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in light_extra.bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=min(5 * 0.5 * total_luminosity**0.6, 5.0),
        )

        extra_mass_models.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=light_extra.bulge, mass=mass)
        )

    extra_galaxies = (
        af.Collection(extra_mass_models) if extra_mass_models else None
    )

    # --- scaling galaxies: fixed light, free shared scaling relation ---
    scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    scaling_relation = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    scaling_mass_models = []
    for i in range(n_scaling):
        light_scaling = light_result.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.Isothermal)
        mass.centre = light_scaling.bulge.centre
        mass.ell_comps = light_scaling.bulge.ell_comps

        luminosity_per_gaussian_list = [
            2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
            for g in light_scaling.bulge.profile_list
        ]
        total_luminosity = np.sum(luminosity_per_gaussian_list) / pixel_scale**2
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

    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
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

Load, plot and mask the `Imaging` data.
"""
dataset_name = "simple"
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

try:
    mask_extra_galaxies = al.Mask2D.from_fits(
        file_path=dataset_path / "mask_extra_galaxies.fits",
        pixel_scales=pixel_scale,
        invert=True,
    )
    dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
except FileNotFoundError:
    pass

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

main_lens_centres.json         — required; determines the number of main lenses.
extra_galaxies_centres.json   — optional; empty list if absent.
scaling_galaxies_centres.json — optional; empty list if absent.

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

positions = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "positions.json")
)

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
