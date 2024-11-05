import autofit as af
import autolens as al

from . import slam_util

from typing import Union, Optional, Tuple


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = af.Model(al.lp.Exponential),
    lens_point: Optional[af.Model] = None,
    mass: af.Model = af.Model(al.mp.Isothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    source_disk: Optional[af.Model] = None,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    mass_centre: Optional[Tuple[float, float]] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. MGE, Sersics).

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of point source emission in the lens galaxy (set to
        None to omit a point).
    mass
        The `MassProfile` fitted by this pipeline.
    shear
        The model used to represent the external shear in the mass model (set to None to turn off shear).
    source_bulge
        The model used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk
        The model used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    redshift_lens
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the SOURCE LP PIPELINE fits a lens model where:

     - The lens galaxy light is modeled using a light profiles [no prior initialization].
     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a light profiles [no prior initialization].

    This search aims to accurately estimate an initial lens light model, mass model and source model.
    """

    if mass_centre is not None:
        mass.centre = mass_centre

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
                point=lens_point,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
                disk=source_disk,
            ),
        ),
        extra_galaxies=extra_galaxies,
        dataset_model=dataset_model,
    )

    """
    For single-dataset analyses, the following code does not change the model or analysis and can be ignored.
    
    For multi-dataset analyses, the following code updates the model and analysis.
    """
    analysis = slam_util.analysis_multi_dataset_from(
        analysis=analysis, model=model, multi_dataset_offset=True
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
    )

    result = search.fit(model=model, analysis=analysis, **settings_search.fit_dict)

    return result
