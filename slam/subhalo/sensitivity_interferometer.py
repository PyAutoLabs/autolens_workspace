import autofit as af
import autolens as al
import autolens.plot as aplt
from autofit.non_linear.grid import sensitivity as s
from typing import Union, Tuple, ClassVar
import numpy as np


def run(
    settings_search: af.SettingsSearch,
    uv_wavelengths: np.ndarray,
    real_space_mask: al.Mask2D,
    mass_result: af.Result,
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
    base_model = mass_result.model

    """
    We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturb_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark mattter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=subhalo_mass)

    """
    Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
    and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
    iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e11, of which only the latter
    will be shown to be detectable.
    """
    perturb_model.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1e6, upper_limit=1e11
    )
    perturb_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturb_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturb_model.mass.redshift_object = mass_result.model.galaxies.lens.redshift
    perturb_model.mass.redshift_source = mass_result.model.galaxies.source.redshift

    """
    We are performing sensitivity mapping to determine when a subhalo is detectable. Eery simulated dataset must 
    be simulated with a lens model, called the `simulation_instance`. We use the maximum likelihood model of the mass pipeline
    for this.

    This includes the lens light and mass and source galaxy light.
    """
    simulation_instance = mass_result.instance

    """
    We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
    simulate a dataset which is subsequently fitted.

    Note that when this dataset is simulated, the quantity `instance.perturb` is used in the `simulate_cls`.
    This is an instance of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called
    based on the value of sensitivity being computed. 

    In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of 
    1e6 MSun and 1e11 MSun.
    """

    def __call__(instance, simulate_path):
        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer(
            galaxies=[
                instance.galaxies.lens,
                instance.perturb,
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
            over_sampling=al.OverSamplingIterate(
                fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
            ),
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
    search = af.Nautilus(
        name="subhalo__sensitivity", **settings_search.search_dict, n_live=100
    )

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
    example it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform 
    sensitivity mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. 
    In this example is composed of an `Isothermal` lens and `Sersic` source.

    - `perturb_model`: This is the extra model component that alongside the `base_model` is fitted to every 
    simulated dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

    - `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the 
    `perturb_model` to simulate many datasets that are fitted with the `base_model` 
    and `base_model` + `perturb_model`.

    - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that 
    fits the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In 
    this example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e11, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e11.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel 
    processing if set above 1.
    """
    sensitivity_mapper = s.Sensitivity(
        search=search,
        simulation_instance=simulation_instance,
        base_model=base_model,
        perturb_model=perturb_model,
        simulate_cls=simulate_cls,
        analysis_class=analysis_cls,
        number_of_steps=number_of_steps,
        number_of_cores=settings_search.number_of_cores,
    )

    return sensitivity_mapper.run()
