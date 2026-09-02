"""
Scaling Relation: SLaM
======================

Uses the SLaM pipelines to fit a lens with two tiers of foreground companions, where the scaling tier's Einstein
radii are tied to the main lens's own Einstein radius by a Faber-Jackson relation.

**This is the script that measures the luminosities.** `modeling.py`, `fit.py` and `likelihood_function.py` all take
the luminosities as given and say they must be measured beforehand; here they are measured, from a light-only fit,
and fed into the relation. That makes this the production path for real data.

A full overview of SLaM is in `guides/modeling/slam_start_here`. This script documents only how it differs.

__Prerequisites__

- **SLaM Start Here** (`guides/modeling/slam_start_here`)
- **Scaling Relation** (`features/scaling_relation/modeling`)
- **Extra Galaxies SLaM** (`features/extra_galaxies/slam`) — the same pipeline with companions modelled
  individually rather than tied.

__Measuring Luminosities__

An MGE light profile's total luminosity is the sum of its Gaussians' luminosities:

    L = sum_k  2 * pi * sigma_k ** 2 / axis_ratio_k * intensity_k   / pixel_scale ** 2

The `intensity` values only exist *after* a fit, because MGE profiles are linear light profiles whose intensities
are solved by linear algebra. So the measurement has to read them off the fitted tracer, via
`max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles`, which converts the solved linear profiles
back into ordinary ones carrying their fitted intensities. `luminosity_from` below does exactly this.

The pipeline measures luminosities twice — once in `lens_light[1]` to set up the masses, and again from `light[1]`
before the final mass model — because the second light fit is better constrained and its luminosities are the ones
the published mass model should rest on.

__The Relation Across Stages__

The anchor's Einstein radius changes profile as the pipeline proceeds, and the tie follows it:

 - `source_lp[1]` — anchor mass is `Isothermal`; the tier ties to its `einstein_radius`.
 - `mass_total[1]` — anchor mass is `PowerLaw`; the tier ties to *that* `einstein_radius` instead.

In both cases the tie multiplies a free model parameter, so the tier stays free of parameters throughout. The
bounded tier is different: a prior's `upper_limit` must be a number, so it needs an *estimate* of the anchor's
Einstein radius — a rough one by hand in the first stage, and the previous stage's fitted value thereafter.

__This Script__

Using LENS LIGHT, SOURCE LP, SOURCE PIX, LIGHT LP and MASS TOTAL pipelines this script fits `Imaging` data where in
the final model:

 - The lens galaxy's light is an MGE bulge and its total mass is a `PowerLaw` plus `ExternalShear`.
 - Each bounded companion has an MGE light profile and a luminosity-bounded free `IsothermalSph` mass.
 - Each scaling companion has a spherical MGE light profile and an `IsothermalSph` mass tied to the anchor.
 - The source galaxy's light is a `Pixelization`.

All companion mass profiles are **untruncated**. Truncation encodes tidal stripping by a host halo's potential and
a galaxy-scale lens has none; the truncated `dPIEMass` form of this tier is the group- and cluster-scale default.

__Contents__

- **Luminosity Measurement:** The MGE luminosity integral used throughout.
- **LENS LIGHT PIPELINE:** Light-only fit of the lens and both tiers — where luminosities come from.
- **SOURCE LP PIPELINE:** Mass and source introduced; the tie applied for the first time.
- **SOURCE PIX PIPELINE 1 / 2:** Pixelized source, tiers carried forward then fixed.
- **LIGHT LP PIPELINE:** A better light model, and re-measured luminosities.
- **MASS TOTAL PIPELINE:** `PowerLaw` anchor, with the tie re-applied to it.
- **Dataset / Centres / Settings / Redshifts / Mesh Shape.**
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

The MGE luminosity integral. `galaxy` must come from a tracer whose linear light profiles have been converted back
to ordinary light profiles, so that each Gaussian carries its fitted `intensity`.
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
                2
                * np.pi
                * gaussian.sigma**2
                / gaussian.axis_ratio()
                * gaussian.intensity
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


def luminosities_from(result, pixel_scale):
    """
    Measure the anchor's luminosity and both tiers' luminosities from a fitted light model.

    The tracer's galaxy list is assembled by the analysis as the `galaxies` collection, then `extra_galaxies`, then
    `scaling_galaxies` (see `autogalaxy/analysis/analysis/analysis.py`). Getting the offsets right matters more than
    it looks: the `galaxies` collection holds the source as well as the lens in every stage after `lens_light[1]`, so
    offsetting the tiers by the number of *lens* galaxies would read the source's profile as a companion's.

    The anchor is `galaxies[0]` because every model in this script composes the lens first.
    """
    tracer = (
        result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    n_galaxies = len(list(result.instance.galaxies))

    n_bounded = (
        len(list(result.instance.extra_galaxies))
        if result.instance.extra_galaxies is not None
        else 0
    )
    n_scaling = (
        len(list(result.instance.scaling_galaxies))
        if result.instance.scaling_galaxies is not None
        else 0
    )

    luminosity_anchor = luminosity_from(
        galaxy=tracer.galaxies[0], pixel_scale=pixel_scale
    )

    bounded_luminosities = [
        luminosity_from(galaxy=tracer.galaxies[n_galaxies + i], pixel_scale=pixel_scale)
        for i in range(n_bounded)
    ]

    scaling_luminosities = [
        luminosity_from(
            galaxy=tracer.galaxies[n_galaxies + n_bounded + i], pixel_scale=pixel_scale
        )
        for i in range(n_scaling)
    ]

    return luminosity_anchor, bounded_luminosities, scaling_luminosities


def bounded_upper_limit_from(
    luminosity, luminosity_anchor, einstein_radius_anchor, cap
):
    """
    The upper bound on a bounded-tier galaxy's Einstein radius: the Faber-Jackson prediction, doubled to stay
    conservative, and capped.
    """
    return min(
        2 * (einstein_radius_anchor / luminosity_anchor**0.5) * luminosity**0.5, cap
    )


"""
__LENS LIGHT PIPELINE__

Not present in `slam_start_here.py`. A light-only fit — no mass, no source — whose sole purpose is to measure a
luminosity for the anchor and every companion. Both tiers get a free MGE with a fixed centre; the scaling tier's is
spherical, which costs no non-linear parameters at all.

Fitting light with no source in the model leaves the lensed arcs unmodelled, which is fine: the arcs are faint
compared to the foreground galaxies whose luminosities we are after, and every later stage refits the light properly.
"""


def lens_light(
    settings_search,
    dataset,
    mask_radius,
    main_lens_centre,
    bounded_galaxies_centres,
    scaling_galaxies_centres,
    redshift_lens,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=main_lens_centre,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    lens = af.Model(
        al.Galaxy, redshift=redshift_lens, bulge=lens_bulge, disk=None, point=None
    )

    bounded_galaxies_list = [
        af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=al.model_util.mge_model_from(
                mask_radius=mask_radius,
                total_gaussians=10,
                centre_fixed=tuple(centre),
                sigma_min=dataset.pixel_scales[0] / 10.0,
            ),
        )
        for centre in bounded_galaxies_centres
    ]

    scaling_galaxies_list = [
        af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=al.model_util.mge_model_from(
                mask_radius=mask_radius,
                total_gaussians=10,
                centre_fixed=tuple(centre),
                use_spherical=True,
                sigma_min=dataset.pixel_scales[0] / 10.0,
            ),
        )
        for centre in scaling_galaxies_centres
    ]

    model = af.Collection(
        galaxies=af.Collection(lens=lens),
        extra_galaxies=af.Collection(bounded_galaxies_list),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    # The scaling tier's spherical fixed-centre MGE contributes NO non-linear parameters — its intensities are solved
    # by linear algebra, which is enough to measure a luminosity from. That is only safe because the lens bulge above
    # is free; a stage in which every other component is a fixed instance would end up with a 0-dimension model,
    # which PyAutoFit rejects. The multi-galaxy sibling pipeline hits exactly that and uses an elliptical MGE instead.
    assert model.prior_count > 0, "lens_light[1] has no free parameters."

    search = af.Nautilus(
        name="lens_light[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE LP PIPELINE__

Equivalent to `source_lp` in `slam_start_here.py`, except the lens light is fixed from `lens_light[1]` and both
companion tiers gain mass here for the first time.

This is where the relation appears. `lens.mass.einstein_radius` is a free parameter of *this* model, so multiplying
it by each scaling galaxy's luminosity ratio ties the tier to it at zero parameter cost. The bounded tier instead
gets a `UniformPrior` whose upper limit uses `einstein_radius_estimate`, because a prior bound has to be a number.
"""


def source_lp(
    settings_search,
    dataset,
    mask_radius,
    lens_light_result,
    luminosity_anchor,
    bounded_galaxies_luminosities,
    scaling_galaxies_luminosities,
    einstein_radius_estimate,
    einstein_radius_cap,
    redshift_lens,
    redshift_source,
    scaling_exponent=0.5,
    n_batch=50,
):
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    lens_instance = lens_light_result.instance.galaxies.lens

    lens = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        bulge=lens_instance.bulge,
        disk=None,
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
    )

    # Bounded tier: light fixed from lens_light[1], mass free inside a luminosity-derived bound.

    bounded_galaxies_list = []

    for i, luminosity in enumerate(bounded_galaxies_luminosities):
        bulge = lens_light_result.instance.extra_galaxies[i].bulge

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = bulge.profile_list[0].centre
        mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=bounded_upper_limit_from(
                luminosity=luminosity,
                luminosity_anchor=luminosity_anchor,
                einstein_radius_anchor=einstein_radius_estimate,
                cap=einstein_radius_cap,
            ),
        )

        bounded_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass)
        )

    # Scaling tier: light fixed from lens_light[1], mass tied to the anchor.

    scaling_galaxies_list = []

    for i, luminosity in enumerate(scaling_galaxies_luminosities):
        bulge = lens_light_result.instance.scaling_galaxies[i].bulge

        mass = af.Model(al.mp.IsothermalSph)
        mass.centre = bulge.profile_list[0].centre
        mass.einstein_radius = (
            lens.mass.einstein_radius
            * (luminosity / luminosity_anchor) ** scaling_exponent
        )

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=mass)
        )

    model = af.Collection(
        galaxies=af.Collection(
            lens=lens,
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        ),
        extra_galaxies=af.Collection(bounded_galaxies_list),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Identical to `slam_start_here.py`, except both tiers are carried forward from `source_lp[1]` as free models. The
scaling tier's tie travels with the model, so it remains tied to the anchor's Einstein radius here without being
re-declared.

__Adapt Image S/N Cap__

The source adapt image is capped at a signal-to-noise of 3.0 before it is used by the adaptive
image-mesh and the adaptive regularization. Without the cap the brightest peak dominates the
weights (they scale as a power of the adapt image), so fainter multiply-imaged features get too
few source pixels and too little regularization weight. Capping makes every feature above S/N 3.0
count equally. The cap is applied to an explicit copy so the raw S/N image is untouched.
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

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

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

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.Isothermal),
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularBilinearAdaptDensity, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        extra_galaxies=source_lp_result.model.extra_galaxies,
        scaling_galaxies=source_lp_result.model.scaling_galaxies,
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Identical to `slam_start_here.py`, except both tiers are fixed as instances from `source_pix[1]`.
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

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=af.Model(al.mesh.RectangularBilinearAdaptImage, shape=mesh_shape),
                    regularization=al.reg.Adapt,
                ),
            ),
        ),
        extra_galaxies=source_pix_result_1.instance.extra_galaxies,
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

Identical to `slam_start_here.py`, except both tiers get a fresh free MGE light profile with their mass fixed from
`source_pix[1]`. This is the light fit whose luminosities the final mass model uses, because it is fitted alongside
a converged source and mass model rather than on its own.
"""


def light_lp(
    settings_search,
    dataset,
    mask_radius,
    main_lens_centre,
    source_result_for_lens,
    source_result_for_source,
    redshift_lens,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=main_lens_centre,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    bounded_galaxies_list = []

    for galaxy in source_result_for_lens.instance.extra_galaxies:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_fixed=tuple(galaxy.mass.centre),
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        bounded_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=galaxy.mass)
        )

    scaling_galaxies_list = []

    for galaxy in source_result_for_lens.instance.scaling_galaxies:
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=10,
            centre_fixed=tuple(galaxy.mass.centre),
            use_spherical=True,
            sigma_min=dataset.pixel_scales[0] / 10.0,
        )

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, mass=galaxy.mass)
        )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
        extra_galaxies=af.Collection(bounded_galaxies_list),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
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

Identical to `slam_start_here.py`, except both tiers keep their `light[1]` light profiles and receive new mass
models built from the **re-measured** `light[1]` luminosities.

The anchor's mass is now a `PowerLaw`, so the tie is re-declared against `PowerLaw.einstein_radius`. The bounded
tier's upper limit no longer needs a hand estimate: `source_pix[1]` fitted the anchor's Einstein radius, so that
value is used instead.
"""


def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    luminosity_anchor,
    bounded_galaxies_luminosities,
    scaling_galaxies_luminosities,
    einstein_radius_anchor,
    redshift_lens,
    scaling_exponent=0.5,
    n_batch=20,
):
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )

    # Cap the source adapt image at S/N 3.0 (see __Adapt Image S/N Cap__ above).
    adapt_image_snr_cap = 3.0

    source_adapt_image = galaxy_image_name_dict["('galaxies', 'source')"].copy()
    source_adapt_image[source_adapt_image > adapt_image_snr_cap] = adapt_image_snr_cap
    galaxy_image_name_dict["('galaxies', 'source')"] = source_adapt_image

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

    mass = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    lens = af.Model(
        al.Galaxy,
        redshift=source_result_for_lens.instance.galaxies.lens.redshift,
        bulge=light_result.instance.galaxies.lens.bulge,
        disk=light_result.instance.galaxies.lens.disk,
        mass=mass,
        shear=source_result_for_lens.model.galaxies.lens.shear,
    )

    source = al.util.chaining.source_from(result=source_result_for_source)

    # Bounded tier: light fixed from light[1]; bound now uses the fitted anchor Einstein radius.

    bounded_galaxies_list = []

    for i, luminosity in enumerate(bounded_galaxies_luminosities):
        light_galaxy = light_result.instance.extra_galaxies[i]

        # Cap at twice the previous stage's fitted radius, so a galaxy cannot grow without limit between stages.
        # Floored at `einstein_radius_floor` because a previous fit that landed near zero would otherwise collapse
        # this into a zero-width prior, which is not a valid prior rather than a tight one.
        cap = max(2.0 * light_galaxy.mass.einstein_radius, einstein_radius_floor)

        tier_mass = af.Model(al.mp.IsothermalSph)
        tier_mass.centre = light_galaxy.mass.centre
        tier_mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=bounded_upper_limit_from(
                luminosity=luminosity,
                luminosity_anchor=luminosity_anchor,
                einstein_radius_anchor=einstein_radius_anchor,
                cap=cap,
            ),
        )

        bounded_galaxies_list.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_galaxy.bulge,
                mass=tier_mass,
            )
        )

    # Scaling tier: light fixed from light[1]; mass tied to the PowerLaw anchor.

    scaling_galaxies_list = []

    for i, luminosity in enumerate(scaling_galaxies_luminosities):
        light_galaxy = light_result.instance.scaling_galaxies[i]

        tier_mass = af.Model(al.mp.IsothermalSph)
        tier_mass.centre = light_galaxy.mass.centre
        tier_mass.einstein_radius = (
            lens.mass.einstein_radius
            * (luminosity / luminosity_anchor) ** scaling_exponent
        )

        scaling_galaxies_list.append(
            af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=light_galaxy.bulge,
                mass=tier_mass,
            )
        )

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        extra_galaxies=af.Collection(bounded_galaxies_list),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
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

Load, plot and mask the `Imaging` data.
"""
dataset_name = "scaling_relation"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/scaling_relation/simulator.py"],
        check=True,
    )

pixel_scale = 0.1

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=pixel_scale,
)

# Read the scale back off the dataset: a capped run (`PYAUTO_SMALL_DATASETS=1`) relabels the data at a
# coarser scale, so the literal above is only true as the `from_fits` argument.
pixel_scale = float(dataset.pixel_scales[0])

mask_radius = 6.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Centres__

One JSON per tier. Unlike `modeling.py`, no luminosities are loaded — this pipeline measures them.
"""
main_lens_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "main_lens_centres.json")
)
bounded_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)
scaling_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "scaling_galaxies_centres.json")
)

main_lens_centre = tuple(list(main_lens_centres)[0])

all_centres = (
    list(main_lens_centres)
    + list(bounded_galaxies_centres)
    + list(scaling_galaxies_centres)
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_search = af.SettingsSearch(
    path_prefix=Path("imaging") / "slam",
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
__Einstein Radius Estimate__

Used only for the bounded tier's prior bounds in the first mass stage, where nothing has been fitted yet. Read it
off the data as the apparent radius of the Einstein ring. It is deliberately crude — it sets a bound, not a value —
and it is replaced by the fitted anchor radius in `mass_total`.
"""
einstein_radius_estimate = 1.6
einstein_radius_cap = 1.5

# Smallest upper bound any bounded-tier prior may have, so a near-zero previous fit cannot produce a
# zero-width (invalid) prior in `mass_total`.
einstein_radius_floor = 0.05

"""
__Mesh Shape__

As discussed in the `features/pixelization/modeling` example, the mesh shape is fixed before modeling.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__SLaM Pipeline__
"""
lens_light_result = lens_light(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    main_lens_centre=main_lens_centre,
    bounded_galaxies_centres=bounded_galaxies_centres,
    scaling_galaxies_centres=scaling_galaxies_centres,
    redshift_lens=redshift_lens,
)

"""
The luminosities the relation needs, measured from the light-only fit.
"""
(
    luminosity_anchor,
    bounded_galaxies_luminosities,
    scaling_galaxies_luminosities,
) = luminosities_from(result=lens_light_result, pixel_scale=pixel_scale)

print(f"Measured anchor luminosity:  {luminosity_anchor}")
print(f"Measured bounded tier:       {bounded_galaxies_luminosities}")
print(f"Measured scaling tier:       {scaling_galaxies_luminosities}")

source_lp_result = source_lp(
    settings_search=settings_search,
    dataset=dataset,
    mask_radius=mask_radius,
    lens_light_result=lens_light_result,
    luminosity_anchor=luminosity_anchor,
    bounded_galaxies_luminosities=bounded_galaxies_luminosities,
    scaling_galaxies_luminosities=scaling_galaxies_luminosities,
    einstein_radius_estimate=einstein_radius_estimate,
    einstein_radius_cap=einstein_radius_cap,
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
    main_lens_centre=main_lens_centre,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    redshift_lens=redshift_lens,
)

"""
Re-measure the luminosities from `light[1]`, whose light model is better constrained than `lens_light[1]`'s, and read
the fitted anchor Einstein radius so the bounded tier no longer needs the hand estimate.
"""
(
    luminosity_anchor,
    bounded_galaxies_luminosities,
    scaling_galaxies_luminosities,
) = luminosities_from(result=light_result, pixel_scale=pixel_scale)

einstein_radius_anchor = source_pix_result_1.instance.galaxies.lens.mass.einstein_radius

print(f"\nRe-measured anchor luminosity: {luminosity_anchor}")
print(f"Fitted anchor einstein_radius: {einstein_radius_anchor}")

mass_result = mass_total(
    settings_search=settings_search,
    dataset=dataset,
    source_result_for_lens=source_pix_result_1,
    source_result_for_source=source_pix_result_2,
    light_result=light_result,
    luminosity_anchor=luminosity_anchor,
    bounded_galaxies_luminosities=bounded_galaxies_luminosities,
    scaling_galaxies_luminosities=scaling_galaxies_luminosities,
    einstein_radius_anchor=einstein_radius_anchor,
    redshift_lens=redshift_lens,
)

"""
__Measured Luminosities__

Write the measured luminosities out in the `y, x, luminosity` CSV schema, so a later `modeling.py`-style fit can
consume them via `al.galaxy_table_from_csv` without re-running the pipeline. They are written under `_measured`
names so the simulator's truth CSVs are left intact — on real data you would drop that suffix.
"""
al.galaxy_table_to_csv(
    centres=[main_lens_centre],
    luminosities=[luminosity_anchor],
    file_path=dataset_path / "main_lens_galaxies_measured.csv",
)

al.galaxy_table_to_csv(
    centres=[tuple(c) for c in bounded_galaxies_centres],
    luminosities=bounded_galaxies_luminosities,
    file_path=dataset_path / "extra_galaxies_measured.csv",
)

al.galaxy_table_to_csv(
    centres=[tuple(c) for c in scaling_galaxies_centres],
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies_measured.csv",
)
