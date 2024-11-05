import autofit as af
import autolens as al

from slam import slam_util
from . import subhalo_util

from typing import Optional, Union, Tuple


def run_1_no_subhalo(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    mass_result: af.Result,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The first SLaM SUBHALO PIPELINE for fitting lens mass models which include a dark matter subhalo.

    This pipeline fits the lens model without a dark matter subhalo, providing the Bayesian evidence which we use to
    perform Bayesian model comparison with the models fitted in the second and third pipelines to determine whether a
    dark matter subhalo is detected.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    mass_result
        The result of the SLaM MASS PIPELINE which ran before this pipeline.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SUBHALO PIPELINE fits a lens model where:

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     
     - The source galaxy's light is parametric or a pixelization depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the MASS PIPELINE. 

    This model will be used to perform Bayesian model comparison with models that include a subhalo, to determine if 
    a subhalo is detected.
    """

    source = al.util.chaining.source_from(
        result=mass_result,
    )

    lens = mass_result.model.galaxies.lens

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        extra_galaxies=al.util.chaining.extra_galaxies_from(
            result=mass_result, mass_as_model=True
        ),
        dataset_model=dataset_model,
    )

    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        source_regularization_result=mass_result,
    )

    search = af.Nautilus(
        name="subhalo[1]",
        **settings_search.search_dict,
        n_live=200,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result


def run_2_grid_search(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    mass_result: af.Result,
    subhalo_result_1: af.Result,
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    free_redshift: bool = False,
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: Union[Tuple[int], int] = 5,
    dataset_model: Optional[af.Model] = None,
) -> af.GridSearchResult:
    """
    The SLaM SUBHALO PIPELINE for fitting lens mass models which include a dark matter subhalo.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    mass_result
        The result of the SLaM MASS PIPELINE which ran before this pipeline.
    subhalo_result_1
        The result of the first SLaM SUBHALO PIPELINE which ran before this pipeline.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    free_redshift
        If `True` the redshift of the subhalo is a free parameter in the second and third searches.
    number_of_steps
        The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
    number_of_cores
        The number of cores used to perform the non-linear search grid search. If 1, each model-fit on the grid is
        performed in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    Search 2 of the SUBHALO PIPELINE we perform a [number_of_steps x number_of_steps] grid search of non-linear
    searches where:

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     
     - The source galaxy's light is parametric or a pixelization depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].

     - The subhalo redshift is fixed to that of the lens galaxy.

     - Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.

     - The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    This search aims to detect a dark matter subhalo.
    """

    subhalo = af.Model(al.Galaxy, mass=subhalo_mass)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    if not free_redshift:
        subhalo.redshift = subhalo_result_1.instance.galaxies.lens.redshift
        subhalo.mass.redshift_object = subhalo_result_1.instance.galaxies.lens.redshift
        search_tag = "search_lens_plane"
    else:
        subhalo.redshift = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=subhalo_result_1.instance.galaxies.source.redshift,
        )
        subhalo.mass.redshift_object = subhalo.redshift
        search_tag = "search_multi_plane"

    subhalo.mass.redshift_source = subhalo_result_1.instance.galaxies.source.redshift

    lens = mass_result.model.galaxies.lens

    source = al.util.chaining.source_from(
        result=mass_result,
    )

    model = af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
        extra_galaxies=al.util.chaining.extra_galaxies_from(
            result=subhalo_result_1, mass_as_model=True
        ),
        dataset_model=dataset_model,
    )

    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        source_regularization_result=mass_result,
    )

    search = af.Nautilus(
        name=f"subhalo[2]_[{search_tag}]",
        **settings_search.search_dict,
        n_live=200,
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
        number_of_cores=1,
    )

    result = subhalo_grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_1,
            model.galaxies.subhalo.mass.centre_0,
        ],
        info=settings_search.info,
    )

    subhalo_util.visualize_subhalo_detect(
        result_no_subhalo=subhalo_result_1,
        result=result,
        analysis=analysis,
        paths=subhalo_grid_search.paths,
    )

    return result


def run_3_subhalo(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    subhalo_result_1: af.Result,
    subhalo_grid_search_result_2: af.GridSearchResult,
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    free_redshift: bool = False,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The SLaM SUBHALO PIPELINE for fitting lens mass models which include a dark matter subhalo.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    mass_result
        The result of the SLaM MASS PIPELINE which ran before this pipeline.
    subhalo_result_1
        The result of the first SLaM SUBHALO PIPELINE which ran before this pipeline.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    subhalo_grid_search_result_2
        The result of the second SLaM SUBHALO PIPELINE grid search which ran before this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    free_redshift
        If `True` the redshift of the subhalo is a free parameter in the second and third searches.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    Search 3 of the SUBHALO PIPELINE we refit the lens and source models above but now including a subhalo, where 
    the subhalo model is initialized from the highest evidence model of the subhalo grid search.

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     
     - The source galaxy's light is parametric or a pixelization depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].

     - The subhalo redshift is fixed to that of the lens galaxy.

     - Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.

     - The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    This search aims to refine the parameter estimates and errors of a dark matter subhalo detected in the grid search
    above.
    """

    subhalo = af.Model(
        al.Galaxy,
        redshift=subhalo_result_1.instance.galaxies.lens.redshift,
        mass=subhalo_mass,
    )

    if not free_redshift:
        subhalo.redshift = subhalo_result_1.instance.galaxies.lens.redshift
        subhalo.mass.redshift_object = subhalo_result_1.instance.galaxies.lens.redshift
        refine_tag = "single_plane_refine"
    else:
        subhalo.redshift = af.UniformPrior(
            lower_limit=0.0,
            upper_limit=subhalo_result_1.instance.galaxies.source.redshift,
        )
        subhalo.mass.redshift_object = subhalo.redshift
        refine_tag = "multi_plane_refine"

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_grid_search_result_2.model_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre

    subhalo.redshift = subhalo_grid_search_result_2.model.galaxies.subhalo.redshift
    subhalo.mass.redshift_object = subhalo.redshift

    model = af.Collection(
        galaxies=af.Collection(
            lens=subhalo_grid_search_result_2.model.galaxies.lens,
            subhalo=subhalo,
            source=subhalo_grid_search_result_2.model.galaxies.source,
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(
            result=subhalo_result_1, mass_as_model=True
        ),
        dataset_model=dataset_model,
    )

    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        source_regularization_result=subhalo_result_1,
    )

    search = af.Nautilus(
        name=f"subhalo[3]_[{refine_tag}]",
        **settings_search.search_dict,
        n_live=600,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result
