"""
Multi Galaxy Scaling Relation: SLaM
==================================

Uses the SLaM pipelines to fit a multi-galaxy lens with a scaling tier whose Einstein radii are tied to the
brightest co-dominant deflector.

**This is the script that measures the luminosities.** `modeling.py`, `fit.py` and `likelihood_function.py` all take
them as given and say they must be measured beforehand; here they are measured, and the measurement is genuinely
awkward at this scale, which is the reason this script exists.

This script documents only how it differs from `multi_galaxy/slam.py`, the multi-galaxy SLaM baseline, which in
turn documents only how *it* differs from `guides/modeling/slam_start_here`.

__Prerequisites__

- **Multi Galaxy SLaM** (`multi_galaxy/slam.py`) — the regime baseline. It establishes the four things every
  multi-galaxy pipeline does: one `lens_i` per deflector built in a loop, the external shear held in its own
  `shear_galaxy`, mass centres fixed in `source_lp[1]` and released in `source_pix[1]`, and `n_live` scaling with
  the deflector count. None of that is re-explained here.
- **SLaM Start Here** (`guides/modeling/slam_start_here`) — what the five stages are for.
- **Multi Galaxy Scaling Relation** (`multi_galaxy/features/scaling_relation/modeling`)
- **Imaging Scaling Relation SLaM** (`imaging/features/scaling_relation/slam`) — the single-anchor version, whose
  luminosity-measurement machinery is identical.

__What Changes From The Baseline__

Two things, and both are about the scaling tier:

1. **Two extra lens-light stages run first.** `lens_light[1]` fits the co-dominant pair on the standard mask and
   `lens_light[2]` fits the tier on an enlarged one, because the luminosities the relation needs have to be
   measured before any mass model can use them. The baseline has no equivalent — it goes straight to
   `source_lp[1]`.
2. **A `scaling_galaxies` collection is carried through every stage**, with its Einstein radii tied to the
   brightest deflector's own free `einstein_radius` rather than freed. The tie travels with the model, so it stays
   anchored without being re-declared.

__The Two-Mask Problem__

`modeling.py` masks at 3.0" and treats the tier as mass-only, because its members sit 5.5-7" out and their light
never enters that fit. But this pipeline has to *measure* their luminosities, and you cannot measure the luminosity
of a galaxy you have masked away.

So it uses two datasets built from the same data:

 - `dataset` — the 3.0" mask. Used for every stage that cares about the lensed source, because a mask that reaches
   7.5" wastes enormous effort on empty sky and, with a pixelized source, invites the mesh to chase noise.
 - `dataset_larger` — a mask wide enough to enclose the tier. Used by exactly one stage, `lens_light[2]`, where the
   tier's light is fitted so its luminosity can be integrated. Nothing else can use it: an `adapt_images` array built
   from a standard-mask fit cannot be applied to a larger-mask analysis (see the LIGHT LP header).

This is exactly the structure of a production multi-galaxy SLaM pipeline. It is the main reason a `slam.py` exists
in this folder at all: the single-search `modeling.py` cannot measure what it needs.

__Measuring Luminosities__

An MGE light profile's total luminosity is the sum of its Gaussians' luminosities:

    L = sum_k  2 * pi * sigma_k ** 2 / axis_ratio_k * intensity_k   / pixel_scale ** 2

The `intensity` values only exist after a fit, because MGE profiles are linear light profiles whose intensities are
solved by linear algebra. The measurement therefore reads them off the fitted tracer via
`max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles`.

The brightest galaxy is then `argmax` over the main lenses' measured luminosities — a measurement, not a naming
convention.

__This Script__

Using LENS LIGHT (two stages), SOURCE LP, SOURCE PIX (two stages), LIGHT LP and MASS TOTAL pipelines this script
fits `Imaging` data where in the final model:

 - Each co-dominant deflector has a free MGE bulge and a `PowerLaw` total mass; the system's `ExternalShear`
   is held in its own `shear_galaxy`.
 - Each scaling galaxy has a free MGE bulge and an `IsothermalSph` mass tied to the brightest galaxy.
 - The source galaxy's light is a `Pixelization`.

All scaling-tier mass profiles are **untruncated**: truncation encodes tidal stripping by a host halo's potential,
which a multi-galaxy lens lacks by definition. Truncated `dPIEMass` members belong to the group and cluster
workflows.

__No Bounded Tier Here__

This dataset has co-dominant deflectors and a faint tied tier, and nothing in between, so there is no
`extra_galaxies` collection. A production pipeline often has all three — see
`imaging/features/scaling_relation/slam.py` for the bounded tier, whose Einstein radii are free inside a
luminosity-derived bound. Adding it here means loading `extra_galaxies_centres.json` and composing that collection
alongside the two below.

__Contents__

- **Luminosity Measurement:** The MGE luminosity integral used throughout.
- **LENS LIGHT PIPELINE 1:** The co-dominant pair's light, on the standard mask.
- **LENS LIGHT PIPELINE 2:** The tier's light, on the enlarged mask — the single source of luminosities here.
- **SOURCE LP PIPELINE:** Mass and source introduced; the tie applied for the first time.
- **SOURCE PIX PIPELINE 1 / 2:** Pixelized source, tier carried forward then fixed.
- **LIGHT LP PIPELINE:** A better light model for the pair, and why it cannot re-measure the tier.
- **MASS TOTAL PIPELINE:** `PowerLaw` deflectors, with the tie re-applied to the brightest galaxy's.
- **Dataset / Two Masks / Centres / Settings.**
- **SLaM Pipeline:** Run the stages in order.
- **Measured Luminosities:** Write them out as CSVs for reuse.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt


"""
__Luminosity Measurement__
"""


def luminosity_from(galaxy, pixel_scale):
    """
    The total luminosity of a galaxy's MGE bulge, summed over its Gaussians.

    Raises if the total is not positive. A zero luminosity is what
    `PYAUTO_TEST_MODE` produces — the light stage returns no usable samples, so every
    Gaussian's `intensity` is zero. The scaling relation would then evaluate
    `(0.0 / 0.0) ** 0.5`, and the resulting NaN surfaces much later and far away
    (an INT_MIN index in the inversion mapper, or a NaN in autofit's identifier hash),
    naming neither luminosity nor test mode. Fail here instead, where the cause is legible.
    """
    luminosity = (
        np.sum(
            [
                2 * np.pi * gaussian.sigma**2 / gaussian.axis_ratio() * gaussian.intensity
                for gaussian in galaxy.bulge.profile_list
            ]
        )
        / pixel_scale**2
    )

    if not luminosity > 0:
        raise ValueError(
            f"Measured luminosity is {luminosity}, but the scaling relation needs a positive "
            f"value: it divides by the anchor's luminosity and takes a square root, so a "
            f"non-positive input yields NaN. The light stage this is measured from produced no "
            f"usable samples — the usual cause is running this script under PYAUTO_TEST_MODE, "
            f"which skips or truncates the search. This script needs a real search; it is listed "
            f"in config/build/no_run.yaml for exactly this reason."
        )

    return luminosity


def luminosities_from(result, n_main, pixel_scale):
    """
    Measure the main lenses' and the scaling tier's luminosities from a fitted light model.

    The tracer's galaxy list is assembled by the analysis as the `galaxies` collection, then `extra_galaxies`, then
    `scaling_galaxies` (see `autogalaxy/analysis/analysis/analysis.py`). The offset for the tier must therefore be the
    size of the whole `galaxies` collection, NOT the number of main lenses — from `source_lp[1]` onwards that
    collection also holds the source, and offsetting by `n_main` would read the source's profile as a scaling
    galaxy's.

    The main lenses are the first `n_main` entries because every model in this script composes `**lens_dict` first.
    """
    tracer = result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles

    n_galaxies = len(list(result.instance.galaxies))

    n_extra = (
        len(list(result.instance.extra_galaxies))
        if getattr(result.instance, "extra_galaxies", None) is not None
        else 0
    )
    n_scaling = (
        len(list(result.instance.scaling_galaxies))
        if result.instance.scaling_galaxies is not None
        else 0
    )

    main_luminosities = [
        luminosity_from(galaxy=tracer.galaxies[i], pixel_scale=pixel_scale)
        for i in range(n_main)
    ]

    scaling_luminosities = [
        luminosity_from(
            galaxy=tracer.galaxies[n_galaxies + n_extra + i], pixel_scale=pixel_scale
        )
        for i in range(n_scaling)
    ]

    return main_luminosities, scaling_luminosities


"""
__LENS LIGHT PIPELINE 1__

Not present in `slam_start_here.py`. A light-only fit of the co-dominant pair on the standard mask — no mass, no
source, no tier. One free MGE per deflector, centred on its known position.
"""


def lens_light_1(
    settings_search,
    dataset,
    mask_radius,
    main_lens_centres,
    redshift_lens,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_dict = {}

    for i, centre in enumerate(main_lens_centres):
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
            centre_sigma=0.1,
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=None, point=None
        )

    model = af.Collection(galaxies=af.Collection(**lens_dict))

    search = af.Nautilus(
        name="lens_light[1]",
        **settings_search.search_dict,
        n_live=100 + 30 * len(lens_dict),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LENS LIGHT PIPELINE 2__

Also not in `slam_start_here.py`, and the stage that motivates this whole script. Fitted on `dataset_larger`, it
fixes the pair's light from `lens_light[1]` and introduces the scaling tier with a free MGE each, so their
luminosities become measurable.

Only the tier is free here, so the search is cheap despite the larger mask.

__Why The Tier's Light Is Elliptical Here__

`modeling.py` can give a scaling galaxy a *spherical* MGE with a fixed centre, which costs zero non-linear
parameters — its Gaussian intensities are solved by linear algebra and its widths are fixed by the basis. That is
what makes the tier free in light as well as mass.

It cannot be done in this stage. Every other component here is a fixed instance, so a zero-parameter tier would
leave the model with no free parameters at all and PyAutoFit rejects it outright: *"Model has no priors! Cannot fit
a 0 dimension model."* The tier is the only thing being fitted, so it has to carry some freedom.

Each member therefore gets an elliptical MGE — free `ell_comps`, two parameters each — with its centre still fixed
to the observed position. Fixing centres matters more at this scale than at galaxy scale: a member allowed to wander
has a co-equal deflector nearby to drift onto. (`mgl_slam_batch`-style production pipelines free the centres too,
with a tight uniform prior; that is a reasonable upgrade once you trust your photometry.)

This is worth internalising as the general rule: the scaling relation makes the tier free in **mass**, but measuring
its **light** costs parameters in whichever stage does the measuring.
"""


def lens_light_2(
    settings_search,
    dataset_larger,
    mask_radius_larger,
    lens_light_result_1,
    scaling_galaxies_centres,
    redshift_lens,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset_larger, use_jax=True)

    n_main = sum(
        1
        for key in vars(lens_light_result_1.instance.galaxies)
        if key.startswith("lens_")
    )

    lens_dict = {}

    for i in range(n_main):
        lens_instance = getattr(lens_light_result_1.instance.galaxies, f"lens_{i}")

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=lens_instance.bulge,
            disk=lens_instance.disk,
            point=lens_instance.point,
        )

    scaling_galaxies_list = []

    for centre in scaling_galaxies_centres:
        # Elliptical (free ell_comps), NOT spherical — see the header. A spherical fixed-centre MGE costs zero
        # non-linear parameters, and since every other component in this stage is a fixed instance that would leave
        # the model with no free parameters at all, which PyAutoFit rejects.
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius_larger,
            total_gaussians=10,
            centre_fixed=tuple(centre),
        )

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge)
        )

    model = af.Collection(
        galaxies=af.Collection(**lens_dict),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    assert model.prior_count > 0, (
        "lens_light[2] has no free parameters. Every main lens is a fixed instance here, so the scaling tier must "
        "carry some freedom — do not make its MGE spherical with a fixed centre in this stage."
    )

    search = af.Nautilus(
        name="lens_light[2]",
        **settings_search.search_dict,
        n_live=100 + 30 * len(scaling_galaxies_list),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE LP PIPELINE__

Equivalent to `source_lp` in `slam_start_here.py`, except all light is fixed from the lens-light stages and mass and
source enter here for the first time.

Each co-dominant deflector gets a free `Isothermal`, and the system's single `ExternalShear` is held in its own
`shear_galaxy` at the system centre, exactly as in `multi_galaxy/slam.py` — the shear describes the tidal field of
everything outside the system, so attaching it to one of several co-dominant galaxies would misrepresent it. The
tier's Einstein radii are tied to the brightest galaxy's free `einstein_radius`, so the tier costs nothing.
"""


def source_lp(
    settings_search,
    dataset,
    mask_radius,
    lens_light_result,
    main_luminosities,
    scaling_luminosities,
    brightest_index,
    redshift_lens,
    redshift_source,
    upper_einstein_radius=3.0,
    scaling_exponent=0.5,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    n_main = len(main_luminosities)
    luminosity_brightest = main_luminosities[brightest_index]
    brightest_key = f"lens_{brightest_index}"

    lens_dict = {}

    for i in range(n_main):
        lens_instance = getattr(lens_light_result.instance.galaxies, f"lens_{i}")

        mass = af.Model(al.mp.Isothermal)
        mass.centre = lens_instance.bulge.profile_list[0].centre
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=upper_einstein_radius
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=lens_instance.bulge,
            disk=lens_instance.disk,
            point=lens_instance.point,
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

    # Scaling tier: light fixed from lens_light[2], mass tied to the brightest galaxy.

    scaling_galaxies_list = []

    for i, luminosity in enumerate(scaling_luminosities):
        bulge = lens_light_result.instance.scaling_galaxies[i].bulge

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = bulge.profile_list[0].centre
        mass.einstein_radius = (
            lens_dict[brightest_key].mass.einstein_radius
            * (luminosity / luminosity_brightest) ** scaling_exponent
        )

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass)
        )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=shear_galaxy,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=150 + 30 * n_main,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `slam_start_here.py`, except the tier is carried forward from `source_lp[1]` as a free model. The tie
travels with the model, so it stays anchored to the brightest galaxy without being re-declared.
"""


def source_pix_1(
    settings_search,
    dataset,
    source_lp_result,
    mesh_shape,
    n_batch=20,
):
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

    n_main = sum(
        1 for key in vars(source_lp_result.instance.galaxies) if key.startswith("lens_")
    )

    lens_dict = {}

    for i in range(n_main):
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
                    mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        scaling_galaxies=source_lp_result.model.scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150 + 50 * (n_main - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`, except the tier is fixed as an instance from `source_pix[1]`.
"""


def source_pix_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    n_main = sum(
        1
        for key in vars(source_pix_result_1.instance.galaxies)
        if key.startswith("lens_")
    )

    lens_dict = {}

    for i in range(n_main):
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
                    mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        scaling_galaxies=source_pix_result_1.instance.scaling_galaxies,
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

Identical to `slam_start_here.py`: each co-dominant deflector gets a fresh free MGE with its mass fixed from
`source_pix[1]`, on the **standard** mask.

__Why This Stage Does Not Re-Measure Luminosities__

The galaxy-scale sibling (`imaging/features/scaling_relation/slam.py`) re-measures its luminosities here, because its
tier sits inside the mask and this later light fit is better constrained. At multi-galaxy scale that is not available,
for two reasons — the second is the important one:

 1. This stage runs on the standard mask, so the tier is outside it. Its light contributes nothing here and cannot be
    measured from this fit at all. Running the stage on `dataset_larger` instead does not work either: the
    `adapt_images` come from `source_pix[1]`, which was fitted on the standard mask, and an adapt image is an array
    defined on *its own* mask. Feeding a 11,304-pixel adapt image into a 68,836-pixel analysis fails with
    `TypeError: mul got incompatible shapes for broadcasting`.

 2. Even if it worked, mixing measurements would be wrong. Only the **ratio** `L_i / L_brightest` enters the
    relation. Take `L_brightest` from this fit and the members' `L_i` from `lens_light[2]` and the ratio is built from
    two different light models on two different masks — a systematic error injected straight into every member's mass.
    The relation needs
    all its luminosities from ONE light fit.

So `lens_light[2]` is the single source of luminosities for this pipeline, and `mass_total` uses them directly. The
tier keeps its `lens_light[2]` light and its `source_pix[1]` mass as fixed instances here.
"""


def light_lp(
    settings_search,
    dataset,
    mask_radius,
    source_result_for_lens,
    source_result_for_source,
    redshift_lens,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    n_main = sum(
        1
        for key in vars(source_result_for_lens.instance.galaxies)
        if key.startswith("lens_")
    )

    lens_dict = {}

    for i in range(n_main):
        lens_instance = getattr(source_result_for_lens.instance.galaxies, f"lens_{i}")

        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=30,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=tuple(lens_instance.mass.centre),
        )

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=lens_instance.redshift,
            bulge=bulge,
            disk=None,
            point=None,
            mass=lens_instance.mass,
        )

    # The tier is carried unchanged: light from `lens_light[2]`, mass from `source_pix[1]`. It lies outside this mask,
    # so it neither gains nor loses anything by being here — but it must stay in the model, because its deflections
    # reach inside the mask.
    scaling_galaxies_list = list(source_result_for_lens.instance.scaling_galaxies)

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.instance.galaxies.shear_galaxy,
            source=source,
        ),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=300 + 100 * (n_main - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

Identical to `slam_start_here.py`, except each deflector's mass becomes a `PowerLaw` and the tier is re-tied to the
brightest galaxy's `PowerLaw.einstein_radius`, using the luminosities re-measured from `light[1]`.

Note the brightest galaxy is re-identified from those luminosities rather than reused from the earlier stage. It will
almost always be the same galaxy; deriving it again keeps the pipeline honest if the improved light model disagrees.
"""


def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    main_luminosities,
    scaling_luminosities,
    brightest_index,
    redshift_lens,
    scaling_exponent=0.5,
    n_batch=20,
):
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

    n_main = len(main_luminosities)
    luminosity_brightest = main_luminosities[brightest_index]
    brightest_key = f"lens_{brightest_index}"

    lens_dict = {}

    for i in range(n_main):
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

    scaling_galaxies_list = []

    for i, luminosity in enumerate(scaling_luminosities):
        light_galaxy = light_result.instance.scaling_galaxies[i]

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = light_galaxy.mass.centre
        mass.einstein_radius = (
            lens_dict[brightest_key].mass.einstein_radius
            * (luminosity / luminosity_brightest) ** scaling_exponent
        )

        scaling_galaxies_list.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_galaxy.bulge,
                mass=mass,
            )
        )

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            **lens_dict,
            shear_galaxy=source_result_for_lens.model.galaxies.shear_galaxy,
            source=source,
        ),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=200 + 100 * (n_main - 1),
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__Dataset__
"""
dataset_name = "scaling_relation"
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
        [sys.executable, "scripts/multi_galaxy/features/scaling_relation/simulator.py"],
        check=True,
    )

pixel_scale = 0.05

dataset_full = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=pixel_scale,
)

"""
__Centres__
"""
main_lens_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "main_lens_centres.json")
)
scaling_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "scaling_galaxies_centres.json")
)

all_centres = list(main_lens_centres) + list(scaling_galaxies_centres)

"""
__Two Masks__

The standard mask for every stage that cares about the lensed source, and an enlarged one for the light stages that
have to reach the tier. The enlarged radius is derived from how far out the tier actually sits, rather than
hard-coded, so adding a more distant member widens the mask automatically.

It is capped just inside the image half-width, because a circular mask larger than the data is not a mask, it is a
crash waiting to happen.
"""
mask_radius = 3.0

galaxy_distances = np.sqrt(
    np.asarray([c[0] for c in all_centres]) ** 2
    + np.asarray([c[1] for c in all_centres]) ** 2
)

image_half_width = 0.5 * min(dataset_full.shape_native) * pixel_scale

mask_radius_larger = min(
    max(mask_radius, float(galaxy_distances.max()) + 0.5), image_half_width - 0.1
)

print(f"Standard mask radius: {mask_radius}")
print(f"Enlarged mask radius: {mask_radius_larger:.2f}")

mask = al.Mask2D.circular(
    shape_native=dataset_full.shape_native,
    pixel_scales=dataset_full.pixel_scales,
    radius=mask_radius,
)

dataset = dataset_full.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=list(main_lens_centres),
    )
)

mask_larger = al.Mask2D.circular(
    shape_native=dataset_full.shape_native,
    pixel_scales=dataset_full.pixel_scales,
    radius=mask_radius_larger,
)

dataset_larger = dataset_full.apply_mask(mask=mask_larger)

dataset_larger = dataset_larger.apply_over_sampling(
    over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset_larger.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=all_centres,
    )
)

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.subplot_imaging_dataset(dataset=dataset_larger)

"""
__Settings AutoFit__
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("multi_galaxy") / "slam",
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

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__
"""
lens_light_result_1 = lens_light_1(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centres=main_lens_centres,
    redshift_lens=redshift_lens,
)

lens_light_result_2 = lens_light_2(
    settings_search=settings_search,
    dataset_larger=dataset_larger,
    mask_radius_larger=mask_radius_larger,
    lens_light_result_1=lens_light_result_1,
    scaling_galaxies_centres=scaling_galaxies_centres,
    redshift_lens=redshift_lens,
)

"""
The luminosities the relation needs, and the brightest galaxy identified from them.
"""
main_luminosities, scaling_luminosities = luminosities_from(
    result=lens_light_result_2,
    n_main=len(list(main_lens_centres)),
    pixel_scale=pixel_scale,
)

brightest_index = int(np.argmax(main_luminosities))

print(f"\nMeasured main lens luminosities: {main_luminosities}")
print(f"Measured scaling tier:           {scaling_luminosities}")
print(f"Brightest galaxy is lens_{brightest_index}")

source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    lens_light_result=lens_light_result_2,
    main_luminosities=main_luminosities,
    scaling_luminosities=scaling_luminosities,
    brightest_index=brightest_index,
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

source_pix_result_1 = source_pix_1(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    mesh_shape=mesh_shape,
)

source_pix_result_2 = source_pix_2(
    settings_search=settings_search,
    dataset=dataset,
    source_lp_result=source_lp_result,
    source_pix_result_1=source_pix_result_1,
    mesh_shape=mesh_shape,
)

light_result = light_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    redshift_lens=redshift_lens,
)

"""
No re-measurement here — see the LIGHT LP header. `lens_light[2]` is this pipeline's single source of luminosities,
and `mass_total` uses those, so every ratio entering the relation comes from one light fit on one mask.
"""

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    main_luminosities=main_luminosities,
    scaling_luminosities=scaling_luminosities,
    brightest_index=brightest_index,
    redshift_lens=redshift_lens,
)

"""
__Measured Luminosities__

Written in the `y, x, luminosity` CSV schema so a later `modeling.py`-style fit can consume them via
`al.galaxy_table_from_csv` without re-running the pipeline. The `_measured` suffix keeps the simulator's truth CSVs
intact; on real data you would drop it.
"""
al.galaxy_table_to_csv(
    centres=[tuple(c) for c in main_lens_centres],
    luminosities=main_luminosities,
    file_path=dataset_path / "main_lens_galaxies_measured.csv",
)

al.galaxy_table_to_csv(
    centres=[tuple(c) for c in scaling_galaxies_centres],
    luminosities=scaling_luminosities,
    file_path=dataset_path / "scaling_galaxies_measured.csv",
)
