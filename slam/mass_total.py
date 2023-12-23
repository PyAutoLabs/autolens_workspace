import autofit as af
import autolens as al


from typing import Union, Optional, Tuple


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_results: af.ResultsCollection,
    light_results: Optional[af.ResultsCollection],
    mass: af.Model = af.Model(al.mp.Isothermal),
    multipole: Optional[af.Model] = None,
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    reset_shear_prior: bool = False,
    end_with_adapt_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE, which fits a lens model with a total mass distribution (e.g. a power-law).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    source_results
        The results of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline.
    light_results
        The results of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    reset_shear_prior
        If `True`, the shear of the mass model is reset to the config priors (e.g. broad uniform). This is useful
        when the mass model changes in a way that adds azimuthal structure (e.g. `PowerLawMultipole`) that the
        shear in ass models in earlier pipelines may have absorbed some of the signal of.
    end_with_adapt_extension
        If `True` an adapt extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images generated in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the MASS TOTAL PIPELINE fits a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or a pixelization depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_results[0].model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    if light_results is None:
        bulge = None
        disk = None
        point = None

    else:
        bulge = light_results.last.instance.galaxies.lens.bulge
        disk = light_results.last.instance.galaxies.lens.disk
        point = light_results.last.instance.galaxies.lens.point

    if not reset_shear_prior:
        shear = source_results[0].model.galaxies.lens.shear
    else:
        shear = al.mp.ExternalShear

    if multipole is not None:
        multipole.centre = mass.centre
        multipole.einstein_radius = mass.einstein_radius
        multipole.slope = mass.slope

    source = al.util.chaining.source_from(
        result=source_results.last,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                point=point,
                mass=mass,
                multipole=multipole,
                shear=shear,
                smbh=smbh,
            ),
            source=source,
        ),
        clumps=al.util.chaining.clumps_from(
            result=source_results[0], mass_as_model=True
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]_light[lp]_mass[total]_source",
        **settings_search.search_dict,
        n_live=150,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    """
    __Adapt Extension__

    The above search may be extended with an adapt search, if the SetupAdapt has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
    """

    if end_with_adapt_extension:
        result_1 = al.util.model.adapt_fit(
            setup_adapt=setup_adapt,
            result=result_1,
            analysis=analysis,
            search_previous=search,
        )

    return af.ResultsCollection([result_1])
