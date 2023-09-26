import autofit as af
import autolens as al
from autofit.non_linear.grid import sensitivity as s
from . import slam_util
from . import extensions

from typing import Union, Tuple, ClassVar
import numpy as np


def detection(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_adapt: al.SetupAdapt,
    mass_results: af.ResultsCollection,
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    free_redshift: bool = False,
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: Union[Tuple[int], int] = 5,
    end_with_stochastic_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SLaM SUBHALO PIPELINE for fitting lens mass models which include a dark matter subhalo.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_adapt
        The setup of the adapt fit.
    mass_results
        The results of the SLaM MASS PIPELINE which ran before this pipeline.
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

    source = slam_util.source_from_result_model_if_parametric(
        result=mass_results.last,
    )

    lens = mass_results.last.model.galaxies.lens
    lens.hyper_galaxy = setup_adapt.hyper_galaxy_lens_from(result=mass_results.last)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        clumps=slam_util.clumps_from(result=mass_results.last, mass_as_model=True),
    )

    search_no_subhalo = af.DynestyStatic(
        name="subhalo[1]_mass[total_refine]", **settings_autofit.search_dict, nlive=100
    )

    result_1 = search_no_subhalo.fit(
        model=model, analysis=analysis, **settings_autofit.fit_dict
    )

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
        subhalo.redshift = result_1.instance.galaxies.lens.redshift
        subhalo.mass.redshift_object = result_1.instance.galaxies.lens.redshift
        search_tag = "search_lens_plane"
        refine_tag = "single_plane_refine"
    else:
        subhalo.redshift = af.UniformPrior(
            lower_limit=0.0, upper_limit=result_1.instance.galaxies.source.redshift
        )
        subhalo.mass.redshift_object = subhalo.redshift
        search_tag = "search_multi_plane"
        refine_tag = "multi_plane_refine"

    subhalo.mass.redshift_source = result_1.instance.galaxies.source.redshift

    source = slam_util.source_from_result_model_if_parametric(
        result=mass_results.last,
    )

    model = af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source),
        clumps=slam_util.clumps_from(result=result_1, mass_as_model=True),
    )

    search = af.DynestyStatic(
        name=f"subhalo[2]_mass[total]_source_subhalo[{search_tag}]",
        **settings_autofit.search_dict_x1_core,
        nlive=50,
        walks=5,
        facc=0.2,
        force_x1_cpu=True,  # ensures parallelizing over grid search works.
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search,
        number_of_steps=number_of_steps,
        number_of_cores=settings_autofit.number_of_cores,
    )

    subhalo_result = subhalo_grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_1,
            model.galaxies.subhalo.mass.centre_0,
        ],
        info=settings_autofit.info,
        parent=search_no_subhalo,
    )

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
        al.Galaxy, redshift=result_1.instance.galaxies.lens.redshift, mass=subhalo_mass
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre = subhalo_result.model_absolute(
        a=1.0
    ).galaxies.subhalo.mass.centre

    subhalo.redshift = subhalo_result.model.galaxies.subhalo.redshift
    subhalo.mass.redshift_object = subhalo.redshift

    model = af.Collection(
        galaxies=af.Collection(
            lens=subhalo_result.model.galaxies.lens,
            subhalo=subhalo,
            source=subhalo_result.model.galaxies.source,
        ),
        clumps=slam_util.clumps_from(result=subhalo_result, mass_as_model=True),
    )

    search = af.DynestyStatic(
        name=f"subhalo[3]_subhalo[{refine_tag}]",
        **settings_autofit.search_dict,
        nlive=100,
    )

    result_3 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    if end_with_stochastic_extension:
        extensions.stochastic_fit(
            result=result_3,
            analysis=analysis,
            search_previous=search,
            **settings_autofit.fit_dict,
        )

    return af.ResultsCollection([result_1, subhalo_result, result_3])


class SimulateImaging:
    def __init__(self, mask, psf):
        self.mask = mask
        self.psf = psf

    def __call__(self, instance):
        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                instance.galaxies.lens,
                instance.perturbation,
                instance.galaxies.source,
            ]
        )

        """
        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be tuned to
        match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2DIterate.uniform(
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=self.psf,
            background_sky_level=0.1,
            add_poisson_noise=True,
        )

        simulated_imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

        """
        The data generated by the simulate function is that which is fitted, so we should apply the mask for the analysis 
        here before we return the simulated data.
        """
        return simulated_dataset.apply_mask(mask=self.mask)


def sensitivity_mapping_imaging(
    settings_autofit: af.SettingsSearch,
    mask: al.Mask2D,
    psf: al.Kernel2D,
    mass_results: af.ResultsCollection,
    analysis_cls: ClassVar[al.AnalysisImaging],
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: Union[Tuple[int], int] = 5,
):
    """
    The SLaM SUBHALO PIPELINE for performing sensitivity mapping, which determines what mass dark matter subhalos
    can be detected where in the dataset.

    Parameters
    ----------
    mask
        The Mask2D that is applied to the imaging data for model-fitting.
    psf
        The Point Spread Function (PSF) used when simulating every image of the strong lens that is fitted by
        sensitivity mapping.
    mass_results
        The results of the SLaM MASS PIPELINE which ran before this pipeline.
    analysis_cls
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit. A
        new instance of this class is created for every model-fit.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    number_of_steps
        The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
    number_of_cores
        The number of cores used to perform the non-linear search grid search. If 1, each model-fit on the grid is
        performed in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
    """

    """
    To begin, we define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is f
    itted to every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model
    which includes one!).
    """

    """
    We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
    every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
    includes one!). 

    For this model, we can use the result of fitting this model to the dataset before sensitivity mapping via the 
    mass pipeline. This ensures the priors associated with each parameter are initialized so as to speed up
    each non-linear search performed during sensitivity mapping.
    """
    base_model = mass_results.last.model

    """
    We now define the `perturbation_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturbation_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark mattter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturbation_model = af.Model(al.Galaxy, redshift=0.5, mass=subhalo_mass)

    """
    Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
    and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
    iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e11, of which only the latter
    will be shown to be detectable.
    """
    perturbation_model.mass.mass_at_200 = af.UniformPrior(
        lower_limit=1e6, upper_limit=1e11
    )
    perturbation_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturbation_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturbation_model.mass.redshift_object = (
        mass_results.last.model.galaxies.lens.redshift
    )
    perturbation_model.mass.redshift_source = (
        mass_results.last.model.galaxies.source.redshift
    )

    """
    We are performing sensitivity mapping to determine when a subhalo is detectable. Eery simulated dataset must 
    be simulated with a lens model, called the `simulation_instance`. We use the maximum likelihood model of the mass pipeline
    for this.

    This includes the lens light and mass and source galaxy light.
    """
    simulation_instance = mass_results.last.instance

    """
    We now write the `simulate_function`, which takes the `simulation_instance` of our model (defined above) and uses it to 
    simulate a dataset which is subsequently fitted.

    Note that when this dataset is simulated, the quantity `instance.perturbation` is used in the `simulate_function`.
    This is an instance of the `NFWMCRLudlowSph`, and it is different every time the `simulate_function` is called
    based on the value of sensitivity being computed. 

    In this example, this `instance.perturbation` corresponds to two different subhalos with values of `mass_at_200` of 
    1e6 MSun and 1e11 MSun.
    """
    simulate_function = SimulateImaging(mask=mask, psf=psf)

    """
    We next specify the search used to perform each model fit by the sensitivity mapper.
    """
    search = af.DynestyStatic(
        name="subhalo__sensitivity", **settings_autofit.search_dict, nlive=50
    )

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
    example it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform 
    sensitivity mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. 
    In this example is composed of an `Isothermal` lens and `Sersic` source.

    - `perturbation_model`: This is the extra model component that alongside the `base_model` is fitted to every 
    simulated dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

    - `simulate_function`: This is the function that uses the `simulation_instance` and many instances of the 
    `perturbation_model` to simulate many datasets that are fitted with the `base_model` 
    and `base_model` + `perturbation_model`.

    - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that 
    fits the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturbation_model` are iterated. In 
    this example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e11, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e11.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel 
    processing if set above 1.
    """
    sensitivity_mapper = s.Sensitivity(
        base_model=base_model,
        perturbation_model=perturbation_model,
        simulation_instance=simulation_instance,
        simulate_function=simulate_function,
        analysis_class=analysis_cls,
        search=search,
        number_of_steps=number_of_steps,
        number_of_cores=settings_autofit.number_of_cores,
    )

    return sensitivity_mapper.run()


def sensitivity_mapping_interferometer(
    settings_autofit: af.SettingsSearch,
    uv_wavelengths: np.ndarray,
    real_space_mask: al.Mask2D,
    mass_results: af.ResultsCollection,
    analysis_cls: ClassVar[al.AnalysisInterferometer],
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: Union[Tuple[int], int] = 5,
):
    """
    The SLaM SUBHALO PIPELINE for performing sensitivity mapping, which determines what mass dark matter subhalos
    can be detected where in the dataset.

    Parameters
    ----------
    path_prefix
        The prefix of folders between the output path and the search folders.
    uv_wavelengths
        The wavelengths of the interferometer baselines used for mapping to Fourier space.
    real_space_mask
        The mask in real space which defines how lensed images are computed.
    mass_results
        The results of the SLaM MASS PIPELINE which ran before this pipeline.
    analysis_cls
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit. A
        new instance of this class is created for every model-fit.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    number_of_steps
        The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
    number_of_cores
        The number of cores used to perform the non-linear search grid search. If 1, each model-fit on the grid is
        performed in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
    unique_tag
        The unique tag for this model-fit, which will be given a unique entry in the sqlite database and also acts as
        the folder after the path prefix and before the search name. This is typically the name of the dataset.
    """

    """
    To begin, we define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is f
    itted to every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model
    which includes one!).
    """

    """
    We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
    every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
    includes one!). 

    For this model, we can use the result of fitting this model to the dataset before sensitivity mapping via the 
    mass pipeline. This ensures the priors associated with each parameter are initialized so as to speed up
    each non-linear search performed during sensitivity mapping.
    """
    base_model = mass_results.last.model

    """
    We now define the `perturbation_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturbation_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark mattter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturbation_model = af.Model(al.Galaxy, redshift=0.5, mass=subhalo_mass)

    """
    Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
    and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
    iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e11, of which only the latter
    will be shown to be detectable.
    """
    perturbation_model.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1e6, upper_limit=1e11
    )
    perturbation_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturbation_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturbation_model.mass.redshift_object = (
        mass_results.last.model.galaxies.lens.redshift
    )
    perturbation_model.mass.redshift_source = (
        mass_results.last.model.galaxies.source.redshift
    )

    """
    We are performing sensitivity mapping to determine when a subhalo is detectable. Eery simulated dataset must 
    be simulated with a lens model, called the `simulation_instance`. We use the maximum likelihood model of the mass pipeline
    for this.

    This includes the lens light and mass and source galaxy light.
    """
    simulation_instance = mass_results.last.instance

    """
    We now write the `simulate_function`, which takes the `simulation_instance` of our model (defined above) and uses it to 
    simulate a dataset which is subsequently fitted.

    Note that when this dataset is simulated, the quantity `instance.perturbation` is used in the `simulate_function`.
    This is an instance of the `NFWMCRLudlowSph`, and it is different every time the `simulate_function` is called
    based on the value of sensitivity being computed. 

    In this example, this `instance.perturbation` corresponds to two different subhalos with values of `mass_at_200` of 
    1e6 MSun and 1e11 MSun.
    """

    def simulate_function(instance):
        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                instance.galaxies.lens,
                instance.perturbation,
                instance.galaxies.source,
            ]
        )

        """
        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be tuned to
        match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2DIterate.uniform(
            shape_native=real_space_mask.shape_native,
            pixel_scales=real_space_mask.pixel_scales,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=uv_wavelengths,
            exposure_time=300.0,
            noise_sigma=0.1,
            transformer_class=al.TransformerNUFFT,
        )

        simulated_dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        """
        The data generated by the simulate function is that which is fitted, so we should apply the mask for the 
        analysis here before we return the simulated data.
        """
        return al.Interferometer(
            data=simulated_dataset.visibilities,
            noise_map=simulated_dataset.noise_map,
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
        )

    """
    We next specify the search used to perform each model fit by the sensitivity mapper.
    """
    search = af.DynestyStatic(
        name="subhalo__sensitivity", **settings_autofit.search_dict, nlive=50
    )

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
    example it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform 
    sensitivity mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. 
    In this example is composed of an `Isothermal` lens and `Sersic` source.

    - `perturbation_model`: This is the extra model component that alongside the `base_model` is fitted to every 
    simulated dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

    - `simulate_function`: This is the function that uses the `simulation_instance` and many instances of the 
    `perturbation_model` to simulate many datasets that are fitted with the `base_model` 
    and `base_model` + `perturbation_model`.

    - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that 
    fits the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturbation_model` are iterated. In 
    this example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e11, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e11.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel 
    processing if set above 1.
    """
    sensitivity_mapper = s.Sensitivity(
        search=search,
        simulation_instance=simulation_instance,
        base_model=base_model,
        perturbation_model=perturbation_model,
        simulate_function=simulate_function,
        analysis_class=analysis_cls,
        number_of_steps=number_of_steps,
        number_of_cores=settings_autofit.number_of_cores,
    )

    return sensitivity_mapper.run()
