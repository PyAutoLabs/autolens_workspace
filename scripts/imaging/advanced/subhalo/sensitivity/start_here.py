"""
Feature: Sensitivity Mapping
============================

Bayesian model comparison allows us to take a dataset, fit it with multiple models and use the Bayesian evidence to
quantify which model objectively gives the best-fit following the principles of Occam's Razor.

However, a complex model may not be favoured by model comparison not because it is the 'wrong' model, but simply
because the dataset being fitted is not of a sufficient quality for the more complex model to be favoured. Sensitivity
mapping allows us to address what quality of data would be needed for the more complex model to be favoured or
alternatively for what sets of model parameter values it would be favoured for data of a given quality.

In order to do this, sensitivity mapping involves us writing a function that uses the model(s) to simulate a dataset.
We then use this function to simulate many datasets, for many different models, and fit each dataset to quantify
how much the change in the model led to a measurable change in the data. This is called computing the sensitivity.

How we compute the sensitivity is chosen by us, the user. In this example, we will perform multiple model-fits
with a nested sampling search, to effectively use Bayesian model comparison to compute the sensitivity. This allows
us to infer how much of a Bayesian evidence increase we should expect for datasets of varying quality and / or models
with different parameters.

For strong lensing, this process is crucial for dark matter substructure detection, as discussed in the following paper:

https://arxiv.org/abs/0903.4752

In subhalo detection, our strong lens modeling informs us of whether there is a dark matter subhalo at a given (y,x)
image-plane location of the strong lens. We determine this by fitting a lens models which include a subhalo. However,
are only able to detect dark matter subhalos with (y,x) locations near the lensed source light, and when they are
massive enough to perturb its light in an observable way.

Subhalo detection analysis therefore does not tell us where we could detect subhalos and of what mass. To know this,
we must perform sensitivity mapping.

Sensitivity mapping is a process where we simulate many thousands of strong lens images. Each simulated image includes a
dark matter subhalo at a given (y,x) coordinate and at a given mass. We fit each simulated dataset twice,
with a lens model which does not include a subhalo and with a lens model that does. If the Bayesian evidence
of the lens model including a subhalo is higher than the model which does not, a subhalo at that (Y,x) location and
mass is therefore detectable. For many simulated datasets, we will find the evidence does not increase when we include
a subhalo in the model-fit, informing us that regions of the image-plane away from the lensed source are not sensitive
to subhalos.

The sensitivity map is performed over a three dimensional (or higher) grid of subhalo y,x and mass. Thus, once
sensitivity mapping is complete, we have a complete map of where in the image-plane subhalos of what mass are
detectable. We can plot 2D plots of this grid to visualize where we are sensitive to dark matter subhalos.

The information provided by a sensitivity map is ultimately required to turn dark matter subhalo detections into
constraints on the dark matter subhalo mass function, which is the primary goal of subhalo detection. Thus, it is
necessary to make statements about the nature of dark matter: cold, warm, fuzzy, or something else entirely?
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Analysis + Model-Fit (Base Search)__

We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to simulate 
many realizations of our dataset with a lens model, called the `simulation_instance`. To get this model, we therefore 
fit the data before performing sensitivity mapping so that we can set the `simulation_instance` as the maximum 
likelihood model.

We perform this fit using the lens model we will use to perform sensitivity mapping, which we call the `base_model`.
"""
base_model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    )
)

search_base = af.Nautilus(
    path_prefix=path.join("imaging", "advanced", "subhalo", "sensitivity"),
    name="sensitivity_mapping_base",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search_base.fit(model=base_model, analysis=analysis)

"""
__Base Model__

We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
includes one!). 

For this model, we can use the `base_model` above, however we will use the result of fitting this model to the dataset
before sensitivity mapping. This ensures the priors associated with each parameter are initialized so as to speed up
each non-linear search performed during sensitivity mapping.
"""
base_model = result.model

"""
__Perturb Model__

We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
`centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
passed to the tracer in the simulate function below.

Many instances of the `perturb_model` are created and used to simulate the many strong lens datasets that we fit. 
However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
determine if the subhalo was detectable.

By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark matter 
subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
therefore when it is detectable!).
"""
perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.NFWMCRLudlowSph)

"""
__Mapping Grid__

Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e13, of which only the latter
will be shown to be detectable.
"""
perturb_model.mass.centre.centre_0 = 1.6
perturb_model.mass.centre.centre_1 = 0.0
perturb_model.mass.redshift_object = 0.5
perturb_model.mass.redshift_source = 1.0
perturb_model.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1e6, upper_limit=1e13)

"""
__Simulation Instance__

We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to 
simulate many realizations of our dataset with a lens model, called the `simulation_instance`. This model uses the
result of the fit above.

The code below ensures that the lens light, mass and source parameters of the strong lens are used when simulating
each dataset with a dark matter subhalo.
"""
simulation_instance = result.instance

"""
__Simulate Function Class__

We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
simulate a strong lens dataset, which include a dark matter subhalo, which is subsequently fitted.

Additional attributes required to simulate the data (mask, PSF) can be passed to the `__init__` method, and the 
simulation is  performed in the `__call__` method.

When this dataset is simulated, the quantity `instance.perturb` is used in `__call__`. This is an instance 
of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called based on the value of sensitivity 
being computed. 

In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of 
1e6 MSun and 1e13 MSun.
"""


class SimulateImaging:
    def __init__(self, mask, psf):
        """
        Class used to simulate the strong lens imaging used for sensitivity mapping.

        Parameters
        ----------
        mask
            The mask applied to the real image data, which is applied to every simulated imaging.
        psf
           The PSF of the real image data, which is applied to every simulated imaging and used for each fit.
        """
        self.mask = mask
        self.psf = psf

    def __call__(self, instance: af.ModelInstance, simulate_path: str):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each strong lens image fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Use the input galaxies of the sensitivity `instance` to set up a tracer, which generates the image-plane
           image of the strong lens system.

        2) Simulate this image using the input dataset noise (Poisson) and PSF.

        3) Apply the mask used in the analysis of the real image to the simulated image.

        4) Output information about the simulation to hard-disk.

        The `subhalo` in the sensitivity `instance` changes for every iteration of the sensitivity mapping, ensuring
        that we map out the sensitivity of the analysis to the subhalo properties (centre, mass, etc.).

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the galaxies whose parameters are varied to perform sensitivity.
            The subhalo in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a strong lens, which id input into the fits of the sensitivity mapper.
        """

        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer.from_galaxies(
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

        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        """
        The data generated by the simulate function is that which is fitted, so we should apply the mask for 
        the analysis here before we return the simulated data.
        """
        dataset = dataset.apply_mask(mask=self.mask)

        """
        Outputs info about the `Tracer` to the fit, so we know exactly how we simulated the image.
        """
        self.output_info(simulate_path=simulate_path, dataset=dataset, tracer=tracer)

        return dataset

    def output_info(self, simulate_path: str, dataset: al.Imaging, tracer: al.Imaging):
        """
        Output information about the data simulated for this iteration of sensitivity mapping.

        This information output is as follows:

        - A subplot of the simulated imaging dataset.
        - A subplot of the tracer used to simulate this imaging dataset.
        - A .json file containing the tracer galaxies.

        Parameters
        ----------
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.
        dataset
            The simulated imaging dataset which is visualized.
        tracer
            The tracer used to simulate the imaging dataset, which is visualized and output to a .json file.
        """

        mat_plot = aplt.MatPlot2D(output=aplt.Output(path=simulate_path, format="png"))

        dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
        dataset_plotter.subplot_dataset()

        tracer_plotter = aplt.TracerPlotter(
            tracer=tracer, grid=dataset.grid, mat_plot_2d=mat_plot
        )
        tracer_plotter.subplot_lensed_images()

        al.output_to_json(
            obj=tracer,
            file_path=os.path.join(simulate_path, "tracer.json"),
        )


"""
__Analysis Class__

We are about to write functions which fit the base and perturb models to the simulated imaging data. 

It is convenient to define a `Analysis` class, which behaves analogously to the `Analysis` class used elsewhere
but ensures the adapt images are set up based on the `perturb` model of the sensitivity instance.
"""


class AnalysisImagingSensitivity(al.AnalysisImaging):
    def __init__(self, dataset):
        # TODO : PRELOADS, need to make sure w_tilde isnt repeated over and over.

        super().__init__(dataset=dataset)

        self.adapt_galaxy_image_path_dict = result.adapt_galaxy_image_path_dict
        self.adapt_model_image = result.adapt_model_image


"""
__Base Fit__

We have defined a `Simulate` class that will be used to simulate every dataset simulated by the sensitivity mapper.
Each simulated dataset will have a unique set of parameters for the `subhalo` (e.g. due to different values of
`perturb_model`.

We will fit each simulated dataset using the `base_model`, which quantifies whether not including the dark matter
subhalo in the model changess the goodness-of-fit and therefore indicates if we are sensitive to the subhalo.

We now write a `BaseFit` class, defining how the `base_model` is fitted to each simulated dataset and the 
goodness-of-fit used to quantify whether the model fits the data well. As above, the `__init__` method can be
extended with new inputs to control how the model is fitted and the `__call__` method performs the fit.

In this example, we use a full non-linear search to fit the `base_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `base_model` fit
to the simulated data.
"""


class BaseFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the dark matter subhalo, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `AnalysisImaging` class used to fit the model
        to the dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


"""
__Perturb Fit__

We now define a `PerturbFit` class, which defines how the `perturb_model` is fitted to each simulated dataset. This
behaves analogously to the `BaseFit` class above, but now fits the `perturb_model` to the simulated data (as
opposed to the `base_model`).

Again, in this example we use a full non-linear search to fit the `perturb_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `perturb_model` fit
to the simulated data.
"""


class PerturbFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the dark matter subhalo, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


"""
We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
object below are:

- `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this example 
it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform sensitivity 
mapping on.

- `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. In 
this example is composed of an `Isothermal` lens and `Sersic` source.

- `perturb_model`: This is the extra model component that alongside the `base_model` is fitted to every simulated 
dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

- `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model` 
to simulate many datasets that are fitted with the `base_model` and `base_model` + `perturb_model`.

- `base_fit_cls`: This is the function that fits the `base_model` to every simulated dataset and returns the
goodness-of-fit of the model to the data.

- `perturb_fit_cls`: This is the function that fits the `base_model` + `perturb_model` to every simulated dataset and
returns the goodness-of-fit of the model to the data.

- `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this 
example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e13, therefore 
the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e13.

- `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
if set above 1.
"""
paths = af.DirectoryPaths(
    path_prefix=path.join("features"),
    name="sensitivity_mapping",
)

sensitivity = af.Sensitivity(
    paths=paths,
    simulation_instance=simulation_instance,
    base_model=base_model,
    perturb_model=perturb_model,
    simulate_cls=SimulateImaging(mask=mask, psf=dataset.psf),
    base_fit_cls=BaseFit(analysis_cls=AnalysisImagingSensitivity),
    perturb_fit_cls=PerturbFit(analysis_cls=AnalysisImagingSensitivity),
    number_of_steps=2,
    number_of_cores=2,
)
sensitivity_result = sensitivity.run()

"""
__Results__

You should now look at the results of the sensitivity mapping in the folder `output/features/sensitivity_mapping`. 

You will note the following 4 model-fits have been performed:

 - The `base_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e6` is included.

 - The `base_model` + `perturb_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e6` 
 is included.

 - The `base_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e13` is included.

 - The `base_model` + `perturb_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e13` is 
 included.

The fit produces a `sensitivity_result`. 

We are still developing the `SensitivityResult` class to provide a data structure that better streamlines the analysis
of results. If you intend to use sensitivity mapping, the best way to interpret the resutls is currently via
**PyAutoFit**'s database and `Aggregator` tools. 
"""
print(sensitivity_result.results[0].result.samples.log_evidence)
print(sensitivity_result.results[1].result.samples.log_evidence)

"""
Finish.
"""
