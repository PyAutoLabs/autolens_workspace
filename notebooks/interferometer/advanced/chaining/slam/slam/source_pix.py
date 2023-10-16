import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union


def run(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    source_lp_results: af.ResultsCollection,
    mesh: af.Model(al.AbstractMesh) = af.Model(al.mesh.DelaunayBrightnessImage),
    regularization: af.Model(al.AbstractRegularization) = af.Model(
        al.reg.AdaptiveBrightnessSplit
    ),
) -> af.ResultsCollection:
    """
    The SLaM SOURCE PIX PIPELINE, which initializes a lens model which uses a pixelized source for the source
    analysis.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    source_lp_results
        The results of the SLaM SOURCE LP PIPELINE which ran before this pipeline.
    pixelization
        The pixelization used by the `Inversion` which fits the source light.
    regularization
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    Search 3 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input mesh and regularization.

    This search aims to estimate values for the mesh and regularization scheme.
    """

    analysis.set_adapt_dataset(result=source_lp_results.last)

    model_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.lens.redshift,
                bulge=source_lp_results.last.instance.galaxies.lens.bulge,
                disk=source_lp_results.last.instance.galaxies.lens.disk,
                mass=source_lp_results.last.instance.galaxies.lens.mass,
                shear=source_lp_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=setup_adapt.hyper_galaxy_lens_from(
                    result=source_lp_results.last
                ),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_results.last.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization, mesh=mesh, regularization=regularization
                ),
            ),
        ),
        clumps=slam_util.clumps_from(result=source_lp_results.last),
    )

    search_1 = af.DynestyStatic(
        name="source_pix[1]_light[fixed]_mass[fixed]_source[pix_init]",
        **settings_autofit.search_dict,
        nlive=50,
        dlogz=10.0,
    )

    result_1 = search_1.fit(
        model=model_1, analysis=analysis, **settings_autofit.fit_dict
    )

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    Search 4 of the SOURCE PIX PIPELINE fits a lens model where:

    - The lens galaxy light is modeled using a light profiles [parameters fixed to result of SOURCE LP PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input mesh and regularization scheme [parameters fixed to the result 
     of search 3].

    This search improves the lens mass model by modeling the source using a `Pixelization` and computes the adapt
    images that are used in search 2.
    """
    mass = slam_util.mass_from(
        mass=source_lp_results.last.model.galaxies.lens.mass,
        result=source_lp_results.last,
        unfix_mass_centre=True,
    )

    model_2 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.lens.redshift,
                bulge=result_1.instance.galaxies.lens.bulge,
                disk=result_1.instance.galaxies.lens.disk,
                mass=mass,
                shear=source_lp_results.last.model.galaxies.lens.shear,
                hyper_galaxy=result_1.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
            ),
        ),
        clumps=slam_util.clumps_from(result=source_lp_results.last),
    )

    search_2 = af.DynestyStatic(
        name="source_pix[2]_light[fixed]_mass[total]_source[pix]",
        **settings_autofit.search_dict,
        nlive=50,
    )

    result_2 = search_2.fit(
        model=model_2, analysis=analysis, **settings_autofit.fit_dict
    )

    """
    __Adapt Extension__

    The above search is extended with an adapt search if the SetupAdapt has one or more of the following inputs:

     - The source is modeled using a pixelization with a regularization scheme.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_2 = extensions.adapt_fit(
        setup_adapt=setup_adapt,
        result=result_2,
        analysis=analysis,
        search_previous=search_2,
    )

    return af.ResultsCollection([result_1, result_2])
