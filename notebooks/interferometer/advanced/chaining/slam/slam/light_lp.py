import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional


def run(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_results: af.ResultsCollection,
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = None,
    end_with_hyper_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SlaM LIGHT LP PIPELINE for fitting imaging data with a lens light component.

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
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the LIGHT LP PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using a parametric / basis bulge + disk [no prior initialization].
     - The lens galaxy mass is modeled using SOURCE PIPELINE's mass distribution [Parameters fixed from SOURCE PIPELINE].
     - The source galaxy's light is modeled using SOURCE PIPELINE's model [Parameters fixed from SOURCE PIPELINE].

    This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible in the
    SOURCE PIPELINE as the mass and source models were not properly initialized.
    """

    """
    If hyper-galaxy noise scaling for the lens is on, it may have scaled the noise to high values in the SOURCE
    PIPELINE (which fitted a simpler lens light model than this pipeline). The new lens light model fitted in this
    pipeline may fit the data better, requiring a reducing level of noise scaling. For this reason, the noise scaling
    normalization is included as a free parameter.
    """
    hyper_galaxy = setup_adapt.hyper_galaxy_lens_from(
        result=source_results.last, noise_factor_is_model=True
    )

    source = slam_util.source_from(result=source_results.last, source_is_model=False)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                mass=source_results.last.instance.galaxies.lens.mass,
                shear=source_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=hyper_galaxy,
            ),
            source=source,
        ),
        clumps=slam_util.clumps_from(result=source_results.last, light_as_model=True),
    )

    search = af.DynestyStatic(
        name="light[1]_light[lp]", **settings_autofit.search_dict, nlive=150
    )

    result_1 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    """
    __Hyper Extension__

    The above search is extended with an adapt search if the SetupAdapt has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """

    if end_with_hyper_extension:
        result_1 = extensions.adapt_fit(
            setup_adapt=setup_adapt,
            result=result_1,
            analysis=analysis,
            search_previous=search,
        )

    return af.ResultsCollection([result_1])
