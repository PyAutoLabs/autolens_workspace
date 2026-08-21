"""
SLaM (Source, Light and Mass): Linear Light Profiles (Multi Galaxy)
===================================================================

The SLaM pipeline for a multi-galaxy lens using a single linear `Sersic` per deflector, instead of the MGE bases
used by `multi_galaxy/slam.py`.

This script documents only how it differs from `multi_galaxy/slam.py`, the multi-galaxy SLaM baseline. Read that
first, and `guides/modeling/slam_start_here` before it.

__Contents__

- **What Changes:** The one difference from `multi_galaxy/slam.py`, and when it is the right choice.
- **Source LP Pipeline:** A linear `Sersic` per deflector, plus mass and source.
- **Source Pix Pipeline 1 & 2:** Unchanged from the baseline apart from the light model carried through.
- **Light LP Pipeline:** A fresh linear `Sersic` per deflector.
- **Mass Total Pipeline:** Each deflector's mass promoted to a `PowerLaw`.
- **Dataset, Centres, Mask:** Set up, identical to the baseline.
- **SLaM Pipeline:** Run the five stages in order.

__Prerequisites__

Read `guides/modeling/slam_start_here` first: it describes what the five SLaM stages are and why they are chained
in this order. This script documents only what differs.

__What Changes__

One thing: `al.model_util.mge_model_from(...)` becomes `af.Model(al.lp_linear.Sersic)`. Every stage, every
deflector. The stage structure, chaining and shear handling are identical to `multi_galaxy/slam.py`.

Both models are built from linear light profiles — an MGE *is* a basis of linear Gaussians — so this is not a
choice about whether intensities are solved linearly. They are, either way. It is a choice about how flexible each
deflector's light model is:

 - **MGE (the baseline):** ~20-30 Gaussians per galaxy, 4-6 free parameters. Captures isophotal twists, radially
   varying ellipticity and disturbed morphology.
 - **Linear Sersic (here):** one profile per galaxy, 6 free parameters. Captures a smooth, symmetric galaxy.

__When To Use This Pipeline__

Rarely, on real multi-galaxy data, and it is worth being blunt about why. The systems this regime describes are
frequently interacting pairs — `multi_galaxy/simulator.py` is modelled on the merging pair SDSS J1011+0143 — and
interacting galaxies are precisely the ones whose light a symmetric Sersic cannot represent. The residuals it
leaves sit on top of the other deflector and the arcs, where, as
`multi_galaxy/features/linear_light_profiles/likelihood_function.py` measures, the two deflectors' intensities are
the most strongly coupled pair in the linear system. A light model that is too rigid does not fail quietly here.

Three cases where it is nonetheless the right pipeline:

 1. **The deflectors really are smooth ellipticals**, well separated and undisturbed.
 2. **You need comparable parameters.** A Sersic's `effective_radius` and `sersic_index` are directly comparable
    to a literature catalogue; an MGE's Gaussian weights are not.
 3. **As a fast first pass**, to check the mass model and the data before committing to the full pipeline.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


def n_main_from(result) -> int:
    """
    The number of co-dominant deflectors in a result's model. Identical to the helper in `multi_galaxy/slam.py`.
    """
    return sum(1 for key in vars(result.instance.galaxies) if key.startswith("lens_"))


def lens_bulge_from(centre, centre_sigma: float = 0.1):
    """
    A linear `Sersic` centred on a deflector's known position.

    This is the one function that differs from `multi_galaxy/slam.py`, where the equivalent call is
    `al.model_util.mge_model_from(...)`. It is factored out so the two stages that compose lens light
    (`source_lp[1]` and `light[1]`) stay in step with each other.
    """
    bulge = af.Model(al.lp_linear.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=centre_sigma)
    bulge.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=centre_sigma)
    return bulge


"""
__SOURCE LP PIPELINE__

Identical to `multi_galaxy/slam.py` apart from the light model: each deflector gets a linear `Sersic` rather than
an MGE basis, with its mass centre fixed to its known position and a capped `einstein_radius` prior.

The source keeps its MGE. There is no reason to make the source less flexible — the argument for a Sersic here is
about the *deflectors'* comparability with catalogues, and the source is not being compared to anything.
"""


def source_lp(
    settings_search: af.SettingsSearch,
    dataset,
    mask_radius: float,
    main_lens_centres,
    redshift_lens: float,
    redshift_source: float,
    upper_einstein_radius: float = 3.0,
    n_batch: int = 50,
) -> af.Result:
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i, centre in enumerate(main_lens_centres):

        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=lens_bulge_from(centre=centre),
            disk=None,
            point=None,
            mass=mass,
        )

    shear_galaxy = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        shear=af.Model(al.mp.ExternalShear),
    )

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=shear_galaxy,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * len(lens_dict),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `multi_galaxy/slam.py`. The deflectors' light is carried forward as fixed instances, so whether those
instances are Sersics or MGEs makes no difference to this stage's code.
"""


def source_pix_1(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}

    for i in range(n_main_from(source_lp_result)):

        lens_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        lens_model = getattr(source_lp_result.model.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.Isothermal),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_instance.bulge,
            disk=lens_instance.disk,
            point=lens_instance.point,
            mass=mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_lp_result.model.galaxies.shear_galaxy,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * (n_main_from(source_lp_result) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `multi_galaxy/slam.py`.

One caution specific to this pipeline. The adapt image used here is the lens-light-subtracted image from
`source_pix[1]`, and that subtraction used the Sersic light model. If the Sersics fit the deflectors poorly, their
residuals are carried into the adapt image and therefore into how the source mesh adapts. The baseline pipeline's
MGEs leave much less behind. If the source reconstruction looks structured in a way that traces the deflectors'
positions rather than the arcs, this is why.
"""


def source_pix_2(
    settings_search: af.SettingsSearch,
    dataset,
    source_lp_result: af.Result,
    source_pix_result_1: af.Result,
    mesh,
    regularization,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    lens_dict = {}

    for i in range(n_main_from(source_pix_result_1)):

        lp_instance = getattr(source_lp_result.instance.galaxies, f"lens_{i}")
        pix_instance = getattr(source_pix_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lp_instance.redshift,
            bulge=lp_instance.bulge,
            disk=lp_instance.disk,
            point=lp_instance.point,
            mass=pix_instance.mass,
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_pix_result_1.instance.galaxies.shear_galaxy,
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
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

Identical to `multi_galaxy/slam.py` apart from the light model: a fresh, free linear `Sersic` per deflector,
centred on that deflector's fitted mass centre, with mass and source held fixed.

The baseline's warning about this stage applies unchanged and matters more here. This is where the two galaxies'
light is finally separated, and it is where a rigid light model does its damage: an under-flexible profile on one
deflector has its residual absorbed by the other, biasing the flux ratio. If you are running this pipeline rather
than the baseline, inspect the residuals around *both* galaxies at this stage before going further.
"""


def light_lp(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_dict = {}

    for i in range(n_main_from(source_result_for_lens)):

        lens_instance = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=lens_bulge_from(centre=tuple(lens_instance.mass.centre)),
            disk=None,
            point=None,
            mass=lens_instance.mass,
        )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.instance.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `multi_galaxy/slam.py`: each deflector's mass is promoted to a `PowerLaw`, with the light fixed from
`light[1]` and the source fixed from `source_pix[2]`.
"""


def mass_total(
    settings_search: af.SettingsSearch,
    dataset,
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: af.Result,
    n_batch: int = 20,
) -> af.Result:
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    lens_dict = {}

    for i in range(n_main_from(source_result_for_lens)):

        lens_model = getattr(source_result_for_lens.model.galaxies, f"lens_{i}")
        light_instance = getattr(light_result.instance.galaxies, f"lens_{i}")

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=lens_model.mass,
            unfix_mass_centre=True,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_model.redshift,
            bulge=light_instance.bulge,
            disk=light_instance.disk,
            point=light_instance.point,
            mass=mass,
        )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150 + 100 * (n_main_from(source_result_for_lens) - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__

The `simple` multi-galaxy dataset, set up exactly as in `multi_galaxy/slam.py`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "multi_galaxy" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
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
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

"""
__Mask & Over Sampling__

The standard 3.0" mask, over-sampled at every deflector.

Over-sampling matters more in this pipeline than in the baseline. A `Sersic` with a free `sersic_index` can become
very steeply peaked, and an under-sampled peak is absorbed by the linear solver as a lower intensity rather than
rejected — see `fit.py` in this folder.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

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
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "features" / "linear_light_profiles" / "slam",
    unique_tag=dataset_name,
    info=None,
    session=None,
)

"""
__Redshifts__
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Mesh Shape__
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh_init = af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape)
regularization_init = al.reg.Adapt

mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape)
regularization = al.reg.Adapt

"""
__SLaM Pipeline__
"""
source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_init=mesh_init,
    regularization_init=regularization_init,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh=mesh,
    regularization=regularization,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
)

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
)

"""
__Result__

The checks from `multi_galaxy/slam.py` apply. Add one: compare each deflector's `sersic_index` and
`effective_radius` against each other and against what you expect for galaxies of this type. A Sersic that has run
to an extreme index is usually not describing a galaxy — it is absorbing something the model cannot represent, and
in a multi-galaxy fit what it absorbs generally comes from its neighbour.
"""
print(mass_result.info)

aplt.subplot_fit_imaging(fit=mass_result.max_log_likelihood_fit)

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/slam.py` — the MGE baseline this pipeline substitutes one model into.
 - `multi_galaxy/features/linear_light_profiles/likelihood_function.py` — why the two deflectors' intensities are
   coupled, measured.
 - `imaging/features/linear_light_profiles/slam.py` — the galaxy-scale version.
"""
