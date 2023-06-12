from typing import Tuple, Optional, Union

import autofit as af
import autolens as al


def set_lens_light_centres(lens, light_centre: Tuple[float, float]):
    """
    Set the (y,x) centre of every light profile in the lens light model to the same input value `light_centre`
    Parameters
    ----------
    lens : af.Model(al.Galaxy)
        The `Galaxy` containing the light models of the distribution of the lens galaxy's bulge and disk.
    light_centre
       If input, the centre of every light model centre is set using this (y,x) value.
    """

    if lens.bulge is not None:
        lens.bulge.centre = light_centre

    if lens.disk is not None:
        lens.disk.centre = light_centre


def set_lens_light_model_centre_priors(
    lens: af.Model, light_centre_gaussian_prior_values: Tuple[float, float]
):
    """
    Set the mean and sigma of every `GaussianPrior` of every light profile in the lens light model to the same value,
    for the y and x coordinates.

    This can be used to specifically customize only the prior on the lens light model centre, given that in many
    datasets this is clearly visible by simply looking at the image itself.

    Parameters
    ----------
    lens : af.Model(al.Galaxy)
        The `Galaxy` containing the light models of the distribution of the lens galaxy's bulge and disk.
    light_centre_gaussian_prior_values : (float, float) or None
       If input, the mean and sigma of every light model centre is set using these values as (mean, sigma).
    """

    mean = light_centre_gaussian_prior_values[0]
    sigma = light_centre_gaussian_prior_values[1]

    if lens.bulge is not None:
        lens.bulge.centre_0 = af.GaussianPrior(mean=mean, sigma=sigma)
        lens.bulge.centre_1 = af.GaussianPrior(mean=mean, sigma=sigma)

    if lens.disk is not None:
        lens.disk.centre_0 = af.GaussianPrior(mean=mean, sigma=sigma)
        lens.disk.centre_1 = af.GaussianPrior(mean=mean, sigma=sigma)


def mass_light_dark_from(
    lmp_model: af.Model(al.lmp.LightMassProfile),
    result_light_component: af.Model,
    result: af.Result,
    einstein_mass_range: Optional[Tuple[float, float]] = None,
    as_instance: bool = False,
) -> Optional[af.Model]:
    """
    Returns an updated version of a `LightMassProfile` model (e.g. a bulge or disk) whose priors are initialized from
    previous results of a `Light` pipeline.

    This function generically links any `LightProfile` to any `LightMassProfile`, pairing parameters which share the
    same path.

    It also allows for an Einstein mass range to be input, such that the `LogUniformPrior` on the mass-to-light
    ratio of the lmp_model-component is set with lower and upper limits that are a multiple of the Einstein mass
    computed in the previous SOURCE PIPELINE. For example, if `einstein_mass_range=[0.01, 5.0]` the mass to light
    ratio will use priors corresponding to values which give Einstein masses 1% and 500% of the estimated Einstein mass.

    Parameters
    ----------
    lmp_model : af.Model(al.lmp.LightMassProfile)
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result_light_component : af.Result
        The `LightProfile` result of the LIGHT PIPELINE used to pass the priors.
    result : af.Result
        The result of the LIGHT PIPELINE used to pass the priors.
    einstein_mass_range : (float, float)
        The values a the estimate of the Einstein Mass in the LIGHT PIPELINE is multiplied by to set the lower and
        upper limits of the profile's mass-to-light ratio.
    as_instance
        If `True` the prior is set up as an instance, else it is set up as a lmp_model component.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose priors are initialized from a previous result.
    """

    if lmp_model is None:
        return lmp_model

    lmp_model.take_attributes(source=result_light_component)

    #   lmp_model = result_light_component.instance.galaxies.lens.bulge

    lmp_model = update_mass_to_light_ratio_prior(
        lmp_model=lmp_model, result=result, einstein_mass_range=einstein_mass_range
    )

    return lmp_model


def update_mass_to_light_ratio_prior(
    lmp_model: af.Model(al.lmp.LightMassProfile),
    result: af.Result,
    einstein_mass_range: Tuple[float, float],
    bins: int = 100,
) -> Optional[af.Model]:
    """
    Updates the mass to light ratio parameter of a `LightMassProfile` lmp_model (e.g. a bulge or disk) such that the
    the `LogUniformPrior` on the mass-to-light ratio of the lmp_model-component is set with lower and upper limits that
    are a multiple of the Einstein mass computed in the previous SOURCE PIPELINE.

    For example, if `einstein_mass_range=[0.01, 5.0]` the mass to light ratio will use priors corresponding to
    values which give Einstein masses 1% and 500% of the estimated Einstein mass.

    Parameters
    ----------
    lmp_model
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result
        The result of the LIGHT PIPELINE used to pass the priors.
    einstein_mass_range
        The values a the estimate of the Einstein Mass in the LIGHT PIPELINE is multiplied by to set the lower and upper
        limits of the profile's mass-to-light ratio.
    bins
        The number of bins used to map a calculated Einstein Mass to that of the `LightMassProfile`.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose mass-to-light ratio prior is set using the input Einstein mass and range.
    """

    if einstein_mass_range is None:
        return lmp_model

    if lmp_model is None:
        return None

    grid = result.max_log_likelihood_fit.grid

    einstein_radius = result.max_log_likelihood_tracer.einstein_radius_from(grid=grid)

    einstein_mass = result.max_log_likelihood_tracer.einstein_mass_angular_from(
        grid=grid
    )

    einstein_mass_lower = einstein_mass_range[0] * einstein_mass
    einstein_mass_upper = einstein_mass_range[1] * einstein_mass

    if isinstance(lmp_model, af.Model):
        instance = lmp_model.instance_from_prior_medians(ignore_prior_limits=True)
    else:
        instance = lmp_model

    if instance.intensity < 0.0:
        raise al.exc.GalaxyException(
            "The intensity of a linear light profile is negative, cannot create model."
        )

    mass_to_light_ratio_lower = instance.normalization_via_mass_angular_from(
        mass_angular=einstein_mass_lower, radius=einstein_radius, bins=bins
    )
    mass_to_light_ratio_upper = instance.normalization_via_mass_angular_from(
        mass_angular=einstein_mass_upper, radius=einstein_radius, bins=bins
    )

    lmp_model.mass_to_light_ratio = af.LogUniformPrior(
        lower_limit=mass_to_light_ratio_lower, upper_limit=mass_to_light_ratio_upper
    )

    return lmp_model


def mass_from(mass, result: af.Result, unfix_mass_centre: bool = False) -> af.Model:
    """
    Returns an updated mass `Model` whose priors are initialized from previous results in a pipeline.

    It includes an option to unfix the input `mass_centre` used in the SOURCE PIPELINE, such that if the `mass_centre`
    were fixed (e.g. to (0.0", 0.0")) it becomes a free parameter in this pipeline.

    This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    results
        The result of a previous SOURCE LP PIPELINE or SOURCE PIX PIPELINE.
    unfix_mass_centre
        If the `mass_centre` was fixed to an input value in a previous pipeline, then `True` will unfix it and make it
        free parameters that are fitted for.

    Returns
    -------
    af.Model(mp.MassProfile)
        The total mass profile whose priors are initialized from a previous result.
    """

    mass.take_attributes(source=result.model.galaxies.lens.mass)

    if unfix_mass_centre and isinstance(mass.centre, tuple):
        centre_tuple = mass.centre

        mass.centre = af.Model(mass.cls).centre

        mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

    return mass


def source_from(result: af.Result, source_is_model: bool = False) -> af.Model:
    """
    Setup the source model using the previous pipeline and search results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default SLaM
    pipelines return parametric sources as a model (give they must be updated to properly compute a new mass
    model) and return inversions as an instance (as they have sufficient flexibility to typically not required
    updating). They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

    Parameters
    ----------
    result : af.Result
        The result of the previous source pipeline.
    setup_adapt
        The setup of the adapt fit.
    source_is_model
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        search result it is loaded from. If `False`, it is an instance of that search's result.
    """

    if not hasattr(result.instance.galaxies.source, "pixelization"):
        if source_is_model:
            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
            )

        else:
            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.instance.galaxies.source.bulge,
                disk=result.instance.galaxies.source.disk,
            )

    if hasattr(result, "adapt"):
        if source_is_model:
            pixelization = af.Model(
                al.Pixelization,
                mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
                regularization=result.adapt.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

        else:
            pixelization = af.Model(
                al.Pixelization,
                mesh=result.adapt.instance.galaxies.source.pixelization.mesh,
                regularization=result.adapt.instance.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

    else:
        if source_is_model:
            pixelization = af.Model(
                al.Pixelization,
                mesh=result.instance.galaxies.source.pixelization.mesh,
                regularization=result.model.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )

        else:
            pixelization = af.Model(
                al.Pixelization,
                mesh=result.instance.galaxies.source.pixelization.mesh,
                regularization=result.instance.galaxies.source.pixelization.regularization,
            )

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=pixelization,
            )


def source_from_result_model_if_parametric(
    result: af.Result,
) -> af.Model:
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source_from` method.

    Parameters
    ----------
    result
        The result of the previous source pipeline.
    setup_adapt
        The setup of the adapt fit.
    """

    # TODO : Should not depend on name of pixelization being "pixelization"

    if hasattr(result.instance.galaxies.source, "pixelization"):
        if result.instance.galaxies.source.pixelization is not None:
            return source_from(result=result, source_is_model=False)
    return source_from(result=result, source_is_model=True)


def clean_clumps_of_adapt_images(clumps):
    for clump in clumps:
        if hasattr(clump, "adapt_model_image"):
            del clump.adapt_model_image

        if hasattr(clump, "adapt_galaxy_image"):
            del clump.adapt_galaxy_image


def clumps_from(
    result: af.Result, light_as_model: bool = False, mass_as_model: bool = False
):
    # ideal API:

    # clumps = result.instance.clumps.as_model((al.LightProfile, al.mp.MassProfile,), fixed="centre", prior_pass=True)

    if mass_as_model:
        clumps = result.instance.clumps.as_model((al.mp.MassProfile,))

        for clump_index in range(len(result.instance.clumps)):
            if hasattr(result.instance.clumps[clump_index], "mass"):
                clumps[clump_index].mass.centre = result.instance.clumps[
                    clump_index
                ].mass.centre
                clumps[clump_index].mass.einstein_radius = result.model.clumps[
                    clump_index
                ].mass.einstein_radius

    elif light_as_model:
        clumps = result.instance.clumps.as_model((al.LightProfile,))

        for clump_index in range(len(result.instance.clumps)):
            clumps[clump_index].light.centre = result.instance.clumps[
                clump_index
            ].light.centre
    #     clumps[clump_index].light.intensity = result.model.clumps[clump_index].light.intensity
    #     clumps[clump_index].light.effective_radius = result.model.clumps[clump_index].light.effective_radius
    #     clumps[clump_index].light.sersic_index = result.model.clumps[clump_index].light.sersic_index

    else:
        clumps = result.instance.clumps.as_model(())

    clean_clumps_of_adapt_images(clumps=clumps)

    return clumps


# TODO : Think about how Rich can full generize these.


def lp_from(
    component: Union[al.LightProfile], fit: Union[al.FitImaging, al.FitInterferometer]
) -> al.LightProfile:
    if isinstance(component, al.lp_linear.LightProfileLinear):
        intensity = fit.linear_light_profile_intensity_dict[component]

        return component.lp_instance_from(intensity=intensity)

    elif isinstance(component, al.lp_basis.Basis):
        light_profile_list = []

        for light_profile in component.light_profile_list:
            intensity = fit.linear_light_profile_intensity_dict[light_profile]

            if isinstance(light_profile, al.lp_linear.LightProfileLinear):
                light_profile_list.append(
                    light_profile.lp_instance_from(intensity=intensity)
                )

            else:
                light_profile_list.append(light_profile)

        #   basis = af.Model(al.lp_basis.Basis, light_profile_list=light_profile_list)

        basis = al.lp_basis.Basis(light_profile_list=light_profile_list)

        return basis

    return component


def lmp_from(
    lp: Union[al.LightProfile, al.lp_linear.LightProfileLinear],
    fit: Union[al.FitImaging, al.FitInterferometer],
) -> al.lmp.LightMassProfile:
    if isinstance(lp, al.lp_linear.LightProfileLinear):
        intensity = fit.linear_light_profile_intensity_dict[lp]

        return lp.lmp_model_from(intensity=intensity)

    return lp
