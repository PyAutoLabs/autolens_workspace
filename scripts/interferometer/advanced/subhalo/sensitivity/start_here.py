# """
# Feature: Sensitivity Mapping
# ============================
#
# Bayesian model comparison allows us to take a dataset, fit it with multiple models and use the Bayesian evidence to
# quantify which model objectively gives the best-fit following the principles of Occam's Razor.
#
# However, a complex model may not be favoured by model comparison not because it is the 'wrong' model, but simply
# because the dataset being fitted is not of a sufficient quality for the more complex model to be favoured. Sensitivity
# mapping addresses what quality of data would be needed for the more complex model to be favoured.
#
# In order to do this, sensitivity mapping involves us writing a function that uses the model(s) to simulate a dataset.
# We then use this function to simulate many datasets, for many different models, and fit each dataset using the same
# model-fitting procedure we used to perform Bayesian model comparison. This allows us to infer how much of a Bayesian
# evidence increase we should expect for datasets of varying quality and / or models with different parameters.
#
# For strong lensing, this process is crucial for dark matter substructure detection, as discussed in the following paper:
#
# https://arxiv.org/abs/0903.4752
#
# In substructure detection, we scan a strong lens dark matter subhalos by fitting a lens models which include a subhalo.
# This tells us whether we successfully did detect a subhalo, but it does not tell us where a subhalo has to be located
# (in relation to the source light) to be detectable, nor does to what masses of subhalo we could actually have made a
# detection.
#
# To answer these questions, we must perform sensitivity mapping, where we simulate many thousands of datasets,
# each of which include a dark matter subhalo at a given (y,x) coordinate at a given mass. We then fit each dataset twice,
# once with a lens model which does not include a subhalo and once with a lens model that does. If the Bayesian evidence
# of the model which includes a subhalo is higher than that which does not, then it means a subhalo was detectable!
# """
# # %matplotlib inline
# # from pyprojroot import here
# # workspace_path = str(here())
# # %cd $workspace_path
# # print(f"Working Directory has been set to `{workspace_path}`")
#
# import numpy as np
# from os import path
# import autofit as af
# import autolens as al
# import autolens.plot as aplt
#
# """
# __Dataset + Masking__
#
# Load the `Interferometer` data, define the visibility and real-space masks and plot them.
# """
# real_space_mask = al.Mask2D.circular(
#     shape_native=(151, 151), pixel_scales=0.05, radius=3.0
# )
#
# dataset_name = "simple"
# dataset_path = path.join("dataset", "interferometer", dataset_name)
#
# dataset = al.Interferometer.from_fits(
#     data_path=path.join(dataset_path, "data.fits"),
#     noise_map_path=path.join(dataset_path, "noise_map.fits"),
#     uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
#     real_space_mask=real_space_mask,
# )
# dataset = dataset.apply_settings(
#     settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
# )
#
# dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
# dataset_plotter.subplot_dataset()
#
#
# """
# We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to simulate
# many realizations of our dataset with a lens model, called the `simulation_instance`. To get this model, we therefore
# fit the data before performing sensitivity mapping so that we can set the `simulation_instance` as the maximum
# likelihood model.
#
# We perform this fit using the lens model we will use to perform sensitivity mapping, which we call the `base_model`.
# """
# base_model = af.Collection(
#     galaxies=af.Collection(
#         lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
#         source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore),
#     )
# )
#
# search_base = af.Nautilus(
#     path_prefix=path.join("interferometer", "misc"),
#     name="sensitivity_mapping_base",
#     unique_tag=dataset_name,
#     n_live=100,
# )
#
# analysis = al.AnalysisInterferometer(dataset=dataset)
#
# result = search_base.fit(model=base_model, analysis=analysis)
#
# """
# We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to
# every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which
# includes one!).
#
# For this model, we can use the `base_model` above, however we will use the result of fitting this model to the dataset
# before sensitivity mapping. This ensures the priors associated with each parameter are initialized so as to speed up
# each non-linear search performed during sensitivity mapping.
# """
# base_model = result.model
#
# """
# We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform
# sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
# `centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
# passed to the tracer in the simulate function below.
#
# Many instances of the `perturb_model` are created and used to simulate the many strong lens datasets that we fit.
# However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter
# subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to
# determine if the subhalo was detectable.
#
# By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every
# dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark matter
# subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
# therefore when it is detectable!).
# """
# perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.NFWMCRLudlowSph)
#
# """
# Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
# and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will
# iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e13, of which only the latter
# will be shown to be detectable.
# """
# perturb_model.mass.centre.centre_0 = 1.6
# perturb_model.mass.centre.centre_1 = 0.0
# perturb_model.mass.redshift_object = 0.5
# perturb_model.mass.redshift_source = 1.0
# perturb_model.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1e6, upper_limit=1e13)
#
# """
# We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to
# simulate many realizations of our dataset with a lens model, called the `simulation_instance`. This model uses the
# result of the fit above.
# """
# simulation_instance = result.instance
#
# """
# We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to
# simulate a dataset which is subsequently fitted.
#
# Note that when this dataset is simulated, the quantity `instance.perturb` is used in the `simulate_cls`.
# This is an instance of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called
# based on the value of sensitivity being computed.
#
# In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of
# 1e6 MSun and 1e13 MSun.
# """
#
#
# def __call__(instance, simulate_path):
#     """
#     Set up the `Tracer` which is used to simulate the strong lens interferometer, which may include the subhalo in
#     addition to the lens and source galaxy.
#     """
#     tracer = al.Tracer(
#         galaxies=[
#             instance.galaxies.lens,
#             instance.perturb,
#             instance.galaxies.source,
#         ]
#     )
#
#     """
#     Set up the grid, uv_wavelengths and simulator settings used to simulate interferometer dataset of the strong lens.
#     These should be tuned to match the S/N and noise properties of the observed data you are performing sensitivity
#     mapping on.
#     """
#     grid = al.Grid2D.uniform(
#         shape_native=real_space_mask.shape_native,
#         pixel_scales=real_space_mask.pixel_scales,
# over_sampling = al.OverSamplingIterate(
#    fractional_accuracy=0.9999,
#    sub_steps=[2, 4, 8, 16]
# )
#     )
#
#     simulator = al.SimulatorInterferometer(
#         uv_wavelengths=dataset.uv_wavelengths,
#         exposure_time=300.0,
#         noise_sigma=0.1,
#         transformer_class=al.TransformerNUFFT,
#     )
#
#     simulated_dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
#
#     """
#     The data generated by the simulate function is that which is fitted, so we should apply the mask for the analysis
#     here before we return the simulated data.
#     """
#     return al.Interferometer(
#         data=simulated_dataset.visibilities,
#         noise_map=simulated_dataset.noise_map,
#         uv_wavelengths=simulated_dataset.uv_wavelengths,
#         real_space_mask=real_space_mask,
#     )
#
#
# """
# Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
# data simulated by the `simulate_cls` for that model.
#
# This requires us to write a wrapper around the PyAutoLens `Analysis` class.
# """
#
#
# class AnalysisInterferometerSensitivity(al.AnalysisInterferometer):
#     def __init__(self, dataset):
#         super().__init__(dataset=dataset)
#
#
# """
# We next specify the search used to perform each model fit by the sensitivity mapper.
# """
# search = af.Nautilus(
#     path_prefix=path.join("interferometer", "misc"),
#     name="sensitivity_mapping",
#     unique_tag=dataset_name,
#     n_live=100,
#     force_x1_cpu=True,  # ensures parallelizing over grid search works.
# )
#
# """
# We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
# object below are:
#
# - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this example
# it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform sensitivity
# mapping on.
#
# - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. In
# this example is composed of an `Isothermal` lens and `Sersic` source.
#
# - `perturb_model`: This is the extra model component that alongside the `base_model` is fitted to every simulated
# dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.
#
# - `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model`
# to simulate many datasets that are fitted with the `base_model` and `base_model` + `perturb_model`.
#
# - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that fits
# the data.
#
# - `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this
# example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e13, therefore
# the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e13.
#
# - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
# if set above 1.
# """
# from autofit.non_linear.grid import sensitivity as s
#
# sensitivity = s.Sensitivity(
#     search=search,
#     simulation_instance=simulation_instance,
#     base_model=base_model,
#     perturb_model=perturb_model,
#     simulate_cls=simulate_cls,
#     analysis_class=AnalysisInterferometerSensitivity,
#     number_of_cores=2,
#     number_of_steps=2,
# )
#
# sensitivity_result = sensitivity.run()
#
# """
# You should now look at the results of the sensitivity mapping in the folder `output/features/sensitivity_mapping`.
#
# You will note the following 4 model-fits have been performed:
#
#  - The `base_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e6` is included.
#
#  - The `base_model` + `perturb_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e6`
#  is included.
#
#  - The `base_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e13` is included.
#
#  - The `base_model` + `perturb_model` is fitted to a simulated dataset where a subhalo with `mass_at_200=1e13` is
#  included.
#
# The fit produces a `sensitivity_result`.
#
# We are still developing the `SensitivityResult` class to provide a data structure that better streamlines the analysis
# of results. If you intend to use sensitivity mapping, the best way to interpret the resutls is currently via
# **PyAutoFit**'s database and `Aggregator` tools.
# """
# print(sensitivity_result.results[0].result.samples.log_evidence)
# print(sensitivity_result.results[1].result.samples.log_evidence)
#
# """
# Finish.
# """
