import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Callable, Union, Optional, Tuple


def run(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
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
    The SlaM SOURCE LP PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
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

    In search 1 of the SOURCE LP PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using a parametric / basis bulge + disk [no prior initialization].
     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a parametric / basis bulge + disk [no prior initialization].

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

    search_1 = af.DynestyStatic(
        name="source_lp[1]_light[lp]_mass[total]_source[lp]",
        **settings_autofit.search_dict,
        nlive=200,
        walks=10,
    )

    result_1 = search_1.fit(
        model=model_1, analysis=analysis, **settings_autofit.fit_dict
    )

    """
    __Hyper Extension__

    The above search is extended with an adapt search if the SetupAdapt has one or more of the following inputs:

     - The background sky is included via `hyper_image_sky`.
     - The background noise is included via the `hyper_background_noise`.
     - The source galaxy includes a `HyperGalaxy` for scaling the noise.
    """
    result_1 = extensions.adapt_fit(
        setup_adapt=setup_adapt,
        result=result_1,
        analysis=analysis,
        search_previous=search_1,
    )

    return af.ResultsCollection([result_1])
