import autofit as af
import autolens as al

from . import slam_util

from typing import Union, Optional, Tuple


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    light_result: Optional[af.Result],
    mass: af.Model = af.Model(al.mp.Isothermal),
    multipole_1: Optional[af.Model] = None,
    multipole_3: Optional[af.Model] = None,
    multipole_4: Optional[af.Model] = None,
    smbh: Optional[af.Model] = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    reset_shear_prior: bool = False,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The SLaM MASS TOTAL PIPELINE, which fits a lens model with a total mass distribution (e.g. a power-law).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline,
        used for initializing model components associated with the source galaxy.
    light_result
        The result of the SLaM LIGHT LP PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    light_linear_to_standard
        If `True`, convert all linear light profiles in the model to standard light profiles, whose `intensity` values
        use the max likelihood result of the LIGHT PIPELINE.
    multipole_1
        Optionally include a first order multipole mass profile component in the mass model.
    multipole_3
        Optionally include a third order multipole mass profile component in the mass model.
    multipole_4
        Optionally include a fourth order multipole mass profile component in the mass model.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    reset_shear_prior
        If `True`, the shear of the mass model is reset to the config priors (e.g. broad uniform). This is useful
        when the mass model changes in a way that adds azimuthal structure (e.g. `PowerLawMultipole`) that the
        shear in ass models in earlier pipelines may have absorbed some of the signal of.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
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
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    if light_result is None:
        bulge = None
        disk = None
        point = None

    else:
        bulge = light_result.instance.galaxies.lens.bulge
        disk = light_result.instance.galaxies.lens.disk
        point = light_result.instance.galaxies.lens.point

    if not reset_shear_prior:
        shear = source_result_for_lens.model.galaxies.lens.shear
    else:
        shear = al.mp.ExternalShear

    if multipole_1 is not None:
        multipole_1.m = 1
        multipole_1.centre = mass.centre
        multipole_1.einstein_radius = mass.einstein_radius
        multipole_1.slope = mass.slope

    if multipole_3 is not None:
        multipole_3.m = 3
        multipole_3.centre = mass.centre
        multipole_3.einstein_radius = mass.einstein_radius
        multipole_3.slope = mass.slope

    if multipole_4 is not None:
        multipole_4.m = 4
        multipole_4.centre = mass.centre
        multipole_4.einstein_radius = mass.einstein_radius
        multipole_4.slope = mass.slope

    source = al.util.chaining.source_from(
        result=source_result_for_source,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                point=point,
                mass=mass,
                multipole_1=multipole_1,
                multipole_3=multipole_3,
                multipole_4=multipole_4,
                shear=shear,
                smbh=smbh,
            ),
            source=source,
        ),
        extra_galaxies=al.util.chaining.extra_galaxies_from(
            result=source_result_for_lens, mass_as_model=True
        ),
        dataset_model=dataset_model,
    )

    """
    For single-dataset analyses, the following code does not change the model or analysis and can be ignored.

    For multi-dataset analyses, the following code updates the model and analysis.
    """
    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis,
        model=model,
        multi_dataset_offset=True,
        source_regularization_result=source_result_for_source,
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result
