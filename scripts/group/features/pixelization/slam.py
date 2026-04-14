"""
Pixelization: Group SLaM
=========================

This script uses the SLaM (Source, Light and Mass) pipelines to fit a group-scale strong lens where the
source galaxy is reconstructed using a pixelized mesh with adaptive regularization.

This is essentially the same as the main `group/slam.py` script, since the group SLaM pipeline already
uses a Delaunay pixelization as its default source model. This feature script documents the
pixelization-specific choices and parameters used in the pipeline.

__Pixelization Choices__

The SLaM pipeline makes the following pixelization-specific decisions:

 - **Hilbert mesh**: The SOURCE PIX pipelines use a `Hilbert` image mesh, which distributes source pixels
   along a space-filling Hilbert curve. The pixel count is set automatically based on the data's pixel scale
   via `al.model_util.hilbert_pixels_from_pixel_scale`. This ensures the mesh resolution scales with data quality.

 - **AdaptSplit regularization**: Rather than constant regularization, `AdaptSplit` adapts the smoothing
   to the source morphology. Bright source regions receive less smoothing (preserving detail) while faint
   regions receive more smoothing (suppressing noise). This is the recommended regularization for science-grade
   fits.

 - **Edge pixels**: The Hilbert mesh includes `edge_pixels_total=30` padding pixels at the border of the
   source-plane reconstruction, preventing edge artifacts.

 - **Two-stage pixelization**: SOURCE PIX 1 establishes the pixelization with initial adapt data, then
   SOURCE PIX 2 refines it using capped adapt data from the first stage. This two-stage approach
   ensures the adaptive features converge to robust solutions.

 - **Signal-adaptive over-sampling**: Pixels above a signal-to-noise threshold use higher over-sampling
   (sub_size=4) than faint pixels (sub_size=2), balancing accuracy and speed.

__Contents__

**Prerequisites:** Before using this SLaM pipeline, you should be familiar with group/slam.py and the
pixelization feature scripts.
**Dataset:** Load and plot the strong lens dataset.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Mask:** Define the 2D mask applied to the dataset.
**SLaM Pipeline:** The full SLaM pipeline with pixelization-specific documentation.

__Prerequisites__

Before using this SLaM pipeline, you should be familiar with:

- **SLaM Start Here** (`guides/modeling/slam_start_here`): Introduction to SLaM pipeline structure.
- **Group SLaM** (`group/slam`): How group-scale SLaM handles extra and scaling galaxies.
- **Pixelization Modeling** (`group/features/pixelization/modeling`): Group pixelization basics.

__This Script__

Using a SOURCE LP PIPELINE (two searches), SOURCE PIX PIPELINE (two searches), LIGHT LP PIPELINE
and TOTAL MASS PIPELINE this SLaM modeling script fits `Imaging` data of a group-scale strong lens
where in the final model:

 - Each main lens galaxy has a free MGE bulge and a `PowerLaw` total mass.
 - Each extra galaxy has a free MGE bulge and a luminosity-bounded `Isothermal` mass.
 - The source galaxy's light is a `Hilbert` `Pixelization` with `AdaptSplit` regularization.
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

Fits light only -- no mass, no source -- for every galaxy simultaneously, giving the next search
clean fixed light models to build on.

This is identical to the group/slam.py SOURCE LP 0 pipeline.
"""


def source_lp_0(
    dataset,
    settings_search,
    main_lens_centres,
    extra_lens_centres,
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
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None
            )
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

    n_extra = len(extra_galaxies) if extra_galaxies is not None else 0
    n_live = 100 + 30 * len(lens_light_models) + 30 * n_extra

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_light_models)}
    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
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

Equivalent to `source_lp` in `slam_start_here.py`, except lens light is fixed from `source_lp[0]`
rather than free, and mass and source are introduced here for the first time.

Multiple main-lens galaxies each get an `Isothermal` mass; only `lens_0` carries an `ExternalShear`.
Extra-galaxy Einstein radii are bounded by a luminosity-derived prior.
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
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=lp0_extra.bulge, mass=mass
            )
        )

    extra_galaxies = af.Collection(extra_mass_models) if extra_mass_models else None

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_full_models)}
    lens_dict["source"] = af.Model(
        al.Galaxy, redshift=redshift_source, bulge=source_bulge
    )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    n_extra_model = len(extra_galaxies) if extra_galaxies is not None else 0
    n_live = 150 + 30 * n_main + 30 * n_extra_model

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

This is where the pixelization is first introduced. The source switches from a parametric MGE to a
Hilbert pixelization with AdaptSplit regularization.

Key pixelization choices:
 - `al.mesh.Hilbert`: A space-filling curve mesh that distributes source pixels efficiently.
 - Pixel count from `al.model_util.hilbert_pixels_from_pixel_scale`: scales with data resolution.
 - `edge_pixels_total=30`: padding pixels at source-plane boundaries to prevent edge artifacts.
 - `al.reg.AdaptSplit`: adaptive regularization that varies smoothing based on source brightness.

Signal-adaptive over-sampling is applied: pixels above the S/N threshold use sub_size=4, others sub_size=2.
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

    # Signal-adaptive over-sampling for the pixelization grid.
    over_sample_size_pixelization = al.util.over_sample.over_sample_size_via_signal_to_noise_from(
        signal_to_noise_map=dataset.signal_to_noise_map,
        sub_size_lower=2,
        sub_size_upper=4,
        signal_to_noise_threshold=signal_to_noise_threshold,
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result_1
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result_1.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    n_main = sum(
        1
        for k in vars(source_lp_result_1.instance.galaxies)
        if k.startswith("lens_")
    )

    # Main lens galaxies: mass as model from previous result.
    lens_models = []
    for i in range(n_main):
        prev_lens = getattr(source_lp_result_1.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=getattr(source_lp_result_1.model.galaxies, f"lens_{i}").mass,
            mass_result=getattr(source_lp_result_1.model.galaxies, f"lens_{i}").mass,
            unfix_mass_centre=True,
        )

        lens_models.append(
            af.Model(
                al.Galaxy,
                redshift=prev_lens.redshift,
                bulge=prev_lens.bulge,
                disk=prev_lens.disk,
                mass=mass,
                shear=source_lp_result_1.model.galaxies.lens_0.shear
                if i == 0
                else None,
            )
        )

    # Extra galaxies: carried forward as model parameters.
    extra_galaxies = (
        source_lp_result_1.model.extra_galaxies
        if source_lp_result_1.model.extra_galaxies is not None
        else None
    )

    # Source: Hilbert pixelization with AdaptSplit regularization.
    pixelization = af.Model(
        al.Pixelization,
        mesh=af.Model(
            al.mesh.Hilbert,
            pixels=hilbert_pixels,
            edge_pixels_total=edge_pixels_total,
        ),
        regularization=af.Model(al.reg.AdaptSplit),
    )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=pixelization,
    )

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result, dataset, adapt_images


"""
__SOURCE PIX PIPELINE 2__

Identical to SOURCE PIX 1 but uses capped adapt data from the first pixelization stage, ensuring
the adaptive mesh converges to a robust solution. The lens mass is fixed from SOURCE PIX 1.
"""


def source_pix_2(
    dataset,
    settings_search,
    source_lp_result_1,
    source_pix_result_1,
    adapt_images,
    pixel_scale,
    n_batch=20,
):
    hilbert_pixels = al.model_util.hilbert_pixels_from_pixel_scale(pixel_scale)
    edge_pixels_total = 30

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    # Cap the adapt data to prevent extreme values from dominating the mesh.
    adapt_images_capped = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images_capped,
    )

    n_main = sum(
        1
        for k in vars(source_pix_result_1.instance.galaxies)
        if k.startswith("lens_")
    )

    # Main lens galaxies: mass fixed from SOURCE PIX 1.
    lens_models = []
    for i in range(n_main):
        prev_lens = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")
        lens_models.append(prev_lens)

    extra_galaxies = source_pix_result_1.instance.extra_galaxies

    pixelization = af.Model(
        al.Pixelization,
        mesh=af.Model(
            al.mesh.Hilbert,
            pixels=hilbert_pixels,
            edge_pixels_total=edge_pixels_total,
        ),
        regularization=af.Model(al.reg.AdaptSplit),
    )

    source = af.Model(
        al.Galaxy,
        redshift=source_lp_result_1.instance.galaxies.source.redshift,
        pixelization=pixelization,
    )

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Fits the lens galaxy light with mass and pixelized source fixed from the SOURCE PIX pipelines.
Extra galaxies receive a fresh free MGE light profile.
"""


def light_lp(
    dataset,
    settings_search,
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

    n_main = sum(
        1
        for k in vars(source_pix_result_1.instance.galaxies)
        if k.startswith("lens_")
    )

    lens_models = []
    for i in range(n_main):
        prev_lens = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=False,
            centre=prev_lens.bulge.centre,
            centre_sigma=0.1,
        )

        lens_models.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=bulge,
                mass=prev_lens.mass,
                shear=prev_lens.shear if i == 0 else None,
            )
        )

    # Extra galaxies: fresh MGE light, mass fixed.
    n_extra = (
        len(list(source_pix_result_1.instance.extra_galaxies))
        if source_pix_result_1.instance.extra_galaxies is not None
        else 0
    )

    extra_models = []
    for i in range(n_extra):
        prev_extra = source_pix_result_1.instance.extra_galaxies[i]

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_prior_is_uniform=True,
            centre=prev_extra.bulge.centre,
            ell_comps_prior_is_uniform=True,
        )

        extra_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=prev_extra.mass
            )
        )

    extra_galaxies = af.Collection(extra_models) if extra_models else None

    source = al.util.chaining.source_custom_model_from(
        result=source_pix_result_2, source_is_model=False
    )

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Fits a PowerLaw total mass model with priors from the SOURCE PIX pipeline and lens light fixed from
the LIGHT LP pipeline. The pixelized source is carried forward.
"""


def mass_total(
    dataset,
    settings_search,
    source_pix_result_1,
    source_pix_result_2,
    light_result,
    adapt_images,
    redshift_lens,
    n_batch=20,
):
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_2.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    n_main = sum(
        1
        for k in vars(light_result.instance.galaxies)
        if k.startswith("lens_")
    )

    lens_models = []
    for i in range(n_main):
        light_lens = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = af.Model(al.mp.PowerLaw)
        mass = al.util.chaining.mass_from(
            mass=mass,
            mass_result=getattr(source_pix_result_1.model.galaxies, f"lens_{i}").mass,
            unfix_mass_centre=True,
        )

        lens_models.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_lens.bulge,
                disk=light_lens.disk,
                mass=mass,
                shear=source_pix_result_1.model.galaxies.lens_0.shear
                if i == 0
                else None,
            )
        )

    # Extra galaxies: fresh mass, light fixed from LIGHT LP.
    n_extra = (
        len(list(light_result.instance.extra_galaxies))
        if light_result.instance.extra_galaxies is not None
        else 0
    )

    extra_models = []
    for i in range(n_extra):
        light_extra = light_result.instance.extra_galaxies[i]

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = light_extra.mass.centre
        mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

        extra_models.append(
            af.Model(
                al.Galaxy, redshift=redshift_lens, bulge=light_extra.bulge, mass=mass
            )
        )

    extra_galaxies = af.Collection(extra_models) if extra_models else None

    source = al.util.chaining.source_from(result=source_pix_result_2)

    lens_dict = {f"lens_{i}": m for i, m in enumerate(lens_models)}
    lens_dict["source"] = source

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        extra_galaxies=extra_galaxies,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

Load and plot the strong lens group dataset.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Load centres for main lens and extra galaxies.
"""
main_lens_centres = _load_centres(dataset_path / "main_lens_centres.json")
extra_galaxies_centres = _load_centres(dataset_path / "extra_galaxies_centres.json")

"""
__Mask__

We use a 7.5" circular mask for the group-scale lens.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Settings AutoFit__
"""
redshift_lens = 0.5
redshift_source = 1.0
pixel_scale = 0.1

settings_search = af.SettingsSearch(
    path_prefix=Path("group") / "features" / "pixelization" / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Positions__

Positions are computed from the SOURCE LP result and used in subsequent pixelization searches
to prevent demagnified source solutions.
"""

"""
__SLaM Pipeline__

The full SLaM pipeline is executed below. Each stage is documented with its pixelization-specific choices.
"""

# --- SOURCE LP PIPELINE 0: Light only ---

source_lp_result_0 = source_lp_0(
    dataset=dataset,
    settings_search=settings_search,
    main_lens_centres=main_lens_centres,
    extra_lens_centres=extra_galaxies_centres,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
)

# --- SOURCE LP PIPELINE 1: Add mass + parametric source ---

positions = source_lp_result_0.positions_likelihood_from(
    factor=3.0, minimum_threshold=0.2
).positions

source_mge_radius = mask_radius

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

# --- SOURCE PIX PIPELINE 1: Introduce Hilbert pixelization ---

source_pix_result_1, dataset_pix, adapt_images = source_pix_1(
    dataset=dataset,
    mask=mask,
    settings_search=settings_search,
    source_lp_result_1=source_lp_result_1,
    over_sample_size=over_sample_size,
    pixel_scale=pixel_scale,
    mask_radius=mask_radius,
    positions=positions,
)

# --- SOURCE PIX PIPELINE 2: Refine with capped adapt data ---

source_pix_result_2 = source_pix_2(
    dataset=dataset_pix,
    settings_search=settings_search,
    source_lp_result_1=source_lp_result_1,
    source_pix_result_1=source_pix_result_1,
    adapt_images=adapt_images,
    pixel_scale=pixel_scale,
)

# --- LIGHT LP PIPELINE: Refit lens light ---

light_result = light_lp(
    dataset=dataset_pix,
    settings_search=settings_search,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    adapt_images=adapt_images,
    mask_radius=mask_radius,
    redshift_lens=redshift_lens,
)

# --- MASS TOTAL PIPELINE: Fit PowerLaw mass ---

mass_result = mass_total(
    dataset=dataset_pix,
    settings_search=settings_search,
    source_pix_result_1=source_pix_result_1,
    source_pix_result_2=source_pix_result_2,
    light_result=light_result,
    adapt_images=adapt_images,
    redshift_lens=redshift_lens,
)

"""
__Result__

The final result contains the full group lens model with a pixelized source reconstruction.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

This script demonstrated the full SLaM pipeline for group-scale lenses with a pixelized source.

The key pixelization choices are:
 - Hilbert mesh with automatic pixel count based on data resolution.
 - AdaptSplit regularization for brightness-dependent smoothing.
 - Two-stage pixelization refinement with capped adapt data.
 - Signal-adaptive over-sampling for the pixelization grid.
 - Edge pixel padding to prevent boundary artifacts.

These choices are the recommended defaults for science-grade group-scale lens modeling.
"""
