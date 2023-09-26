import autofit as af
import autolens as al


from typing import Union, Optional


def run(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_results: af.ResultsCollection,
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = None,
    lens_point: Optional[af.Model] = None,
    end_with_adapt_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SlaM LIGHT LP PIPELINE, which fits a complex model for a lens galaxy's light with the mass and source models fixed..

    Parameters
    ----------
    settings_autofit
        A collection of settings that control the behaviour of PyAutoFit thoughout the pipeline (e.g. paths, database,
        parallelization, etc.).
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    source_results
        The results of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of the lens galaxy's point-source(s)
        emission (e.g. a nuclear star burst region) or compact central structures (e.g. an unresolved bulge).
    end_with_adapt_extension
        If `True` an adapt extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images generated in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the LIGHT LP PIPELINE fits a lens model where:

     - The lens galaxy light is modeled using a light profiles [no prior initialization].
     - The lens galaxy mass is modeled using SOURCE PIPELINE's mass distribution [Parameters fixed from SOURCE PIPELINE].
     - The source galaxy's light is modeled using SOURCE PIPELINE's model [Parameters fixed from SOURCE PIPELINE].

    This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible in the
    SOURCE PIPELINE as the mass and source models were not properly initialized.
    """

    source = al.util.chaining.source_custom_model_from(
        result=source_results.last, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                point=lens_point,
                mass=source_results[0].instance.galaxies.lens.mass,
                shear=source_results[0].instance.galaxies.lens.shear,
            ),
            source=source,
        ),
        clumps=al.util.chaining.clumps_from(
            result=source_results[0], light_as_model=True
        ),
    )

    search = af.Nautilus(
        name="light[1]_light[lp]",
        **settings_autofit.search_dict,
        n_live=150,
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search is extended with an adapt search if the SetupAdapt has one or more of the following inputs:

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
