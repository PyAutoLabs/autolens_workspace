import autofit as af
import autolens as al


from typing import Callable, Union, Optional, Tuple


def run(
    settings_search: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = af.Model(al.lp.Exponential),
    mass: af.Model = af.Model(al.mp.Isothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    source_disk: Optional[af.Model] = None,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    mass_centre: Optional[Tuple[float, float]] = None,
    clump_model: Union[al.ClumpModel, al.ClumpModelDisabled] = al.ClumpModelDisabled(),
) -> af.ResultsCollection:
    """
    The SlaM SOURCE LP PIPELINE, which provides an initial model for the lens's light, mass and source using a
    parametric source model (e.g. Sersics, an MGE).

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
    clump_model
        Add additional clumps containing light and mass profiles to the lens model. These have a known input centre and
        are used to model nearby line of sight galaxies.
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

    model_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
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
        clumps=clump_model.clumps,
    )

    search_1 = af.Nautilus(
        name="source_lp[1]_light[lp]_mass[total]_source[lp]",
        **settings_search.search_dict,
        n_live=200,
    )

    result_1 = search_1.fit(
        model=model_1, analysis=analysis, **settings_search.fit_dict
    )

    return af.ResultsCollection([result_1])
