"""
Results: Samples
================

After a non-linear search has completed, it returns a `Result` object that contains information on samples of
the non-linear search, such as the maximum likelihood model instance, the errors on each parameter and the 
Bayesian evidence.

This script illustrates how to use the result to inspect the non-linear search samples.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Model Fit__

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

The code below performs a model-fit using Nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore, disk=None
        ),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
__Plot__

We now have the `Result` object we will cover in this script. 

As a reminder, in the `modeling` scripts we use the `max_log_likelihood_tracer` and `max_log_likelihood_fit` to plot 
the results of the fit.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=mask.derive_grid.all_false
)
tracer_plotter.subplot_tracer()
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Results tutorials `tracer.py` and `fit.py` expand on the `max_log_likelihood_tracer` and `max_log_likelihood_fit`, 
showing how  they can be used to inspect many aspects of a model.

__Samples__

The result contains a `Samples` object, which contains all samples of the non-linear search.

Each sample corresponds to a set of model parameters that were evaluated and accepted by the non linear search, 
in this example `Nautilus`. 

This includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the nested sampling algorithm Nautilus, so the `Samples` object returned is a `SamplesNest` object.
"""
samples = result.samples

print("Nest Samples: \n")
print(samples)

"""
__Parameters__

The parameters are stored as a list of lists, where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
print("All parameters of the very first sample")
print(samples.parameter_lists[0])
print("The fourth parameter of the tenth sample")
print(samples.parameter_lists[9][3])

"""
__Figures of Merit__

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, where:

- The `log_likelihood` is the value evaluated in the `log_likelihood_function`.

- The `log_prior` encodes information on how parameter priors map log likelihood values to log posterior values.

- The `log_posterior` is `log_likelihood + log_prior`.

- The `weight` gives information on how samples are combined to estimate the posterior, which depends on type of search
  used (for `Nautilus` they are all non-zero values which sum to 1).

Lets inspect these values for the tenth sample.
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Instances__

Many results can be returned as an instance of the model, using the Python class structure of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.
"""
instance = samples.max_log_likelihood()
print("Maximum Log Likelihood Model Instance: \n")
print(instance, "\n")

"""
The attributes of the `instance` (e.g. `galaxies`, `lens`) have these names due to how we composed the `Galaxy` and
its light and mass profiles via the `Collection` and `Model` above. 
"""
print(instance.galaxies)

"""
These galaxies will be named according to the model fitted by the search (in this case, `lens` and `source`).
"""
print(instance.galaxies.lens)
print(instance.galaxies.source)

"""
Their light profiles are also named according to model composition allowing individual parameters to be printed.
"""
print(instance.galaxies.lens.mass.einstein_radius)

"""
We can use this list of galaxies to create the maximum log likelihood `Tracer`, which is the property of the result 
we've used up to now!

Using this tracer is expanded upon in the `tracer.py` results tutorial.

(If we had the `Imaging` available we could easily use this to create the maximum log likelihood `FitImaging`).
"""
max_lh_tracer = al.Tracer(galaxies=instance.galaxies)

print(max_lh_tracer)
print(mask.derive_grid.all_false)

# Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
fit = al.FitImaging(dataset=dataset, tracer=max_lh_tracer)
max_lh_tracer = fit.tracer_linear_light_profiles_to_light_profiles

tracer_plotter = aplt.TracerPlotter(
    tracer=max_lh_tracer, grid=mask.derive_grid.all_false
)
tracer_plotter.subplot_tracer()

"""
__Posterior / PDF__

The result contains the full posterior information of our non-linear search, which can be used for parameter 
estimation. 

PDF stands for "Probability Density Function" and it quantifies probability of each model parameter having values
that are sampled. It therefore enables error estimation via a process called marginalization.

The median pdf vector is available, which estimates every parameter via 1D marginalization of their PDFs.
"""
instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(instance, "\n")
print(instance.galaxies.source.bulge)
print()

vector = samples.median_pdf(as_instance=False)

print("Median PDF Model Parameter Lists: \n")
print(vector, "\n")

"""
__Errors__

Methods for computing error estimates on all parameters are provided. 

This again uses 1D marginalization, now at an input sigma confidence limit. 

By inputting `sigma=3.0` margnialization find the values spanning 99.7% of 1D PDF. Changing this to `sigma=1.0`
would give the errors at the 68.3% confidence limit.
"""
instance_upper_sigma = samples.values_at_upper_sigma(sigma=3.0)
instance_lower_sigma = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(instance_upper_sigma.galaxies.source.bulge, "\n")
print(instance_lower_sigma.galaxies.source.bulge, "\n")

"""
They can also be returned at the values of the parameters at their error values.
"""
instance_upper_values = samples.errors_at_upper_sigma(sigma=3.0)
instance_lower_values = samples.errors_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(instance_upper_values.galaxies.source.bulge, "\n")
print(instance_lower_values.galaxies.source.bulge, "\n")

"""
__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any model -- below we create an instance of the last accepted model.
"""
instance = samples.from_sample_index(sample_index=-1)

print(instance.galaxies.lens.mass)
print(instance.galaxies.lens.mass)

"""
__Search Plots__

The Probability Density Functions (PDF's) of the results can be plotted using the non-linear search in-built 
visualization tools.

This fit used `Nautilus` therefore we use the `NestPlotter` for visualization, which wraps in-built
visualization tools.

The `autofit_workspace/*/plots` folder illustrates other packages that can be used to make these plots using
the standard output results formats (e.g. `GetDist.py`).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
__Maximum Likelihood__

The maximum log likelihood value of the model-fit can be estimated by simple taking the maximum of all log
likelihoods of the samples.

If different models are fitted to the same dataset, this value can be compared to determine which model provides
the best fit (e.g. which model has the highest maximum likelihood)?
"""
print("Maximum Log Likelihood: \n")
print(max(samples.log_likelihood_list))

"""
__Bayesian Evidence__

Nested sampling algorithms like Nautilus also estimate the Bayesian evidence (estimated via the nested sampling 
algorithm).

The Bayesian evidence accounts for "Occam's Razor", whereby it penalizes models for being more complex (e.g. if a model
has more parameters it needs to fit the da

The Bayesian evidence is a better quantity to use to compare models, because it penalizes models with more parameters
for being more complex ("Occam's Razor"). Comparisons using the maximum likelihood value do not account for this and
therefore may unjustly favour more complex models.

Using the Bayesian evidence for model comparison is well documented on the internet, for example the following
wikipedia page: https://en.wikipedia.org/wiki/Bayes_factor
"""
print("Maximum Log Likelihood and Log Evidence: \n")
print(samples.log_evidence)

"""
__Lists__

All results can alternatively be returned as a 1D list of values, by passing `as_instance=False`:
"""
max_lh_list = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_list, "\n\n")

"""
The list above does not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:

 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models they are important for tracking the parameters of the model.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
All the methods above are available as lists.
"""
instance = samples.median_pdf(as_instance=False)
values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)

"""
__Latex__

If you are writing modeling results up in a paper, you can use inbuilt latex tools to create latex table code which 
you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your model
uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.
"""
latex = af.text.Samples.latex(
    samples=result.samples,
    median_pdf_model=True,
    sigma=3.0,
    name_to_label=True,
    include_name=True,
    include_quickmath=True,
    prefix="Example Prefix ",
    suffix=r"\\[-2pt]",
)

print(latex)

"""
__Derived Errors (Advanced)__

Computing the errors of a quantity like the `einstein_radius` is simple, because it is sampled by the non-linear 
search. Errors are accessible using the `Samples` object's `errors_from` methods, which marginalize over the 
parameters via the 1D Probability Density Function (PDF).

Computing errors on derived quantities is more tricky, because they are not sampled directly by the non-linear search. 
For example, what if we want the error on the axis-ratio of the mass model? In order to do this we need to create the 
PDF of that derived quantity, which we can then marginalize over using the same function we use to marginalize model 
parameters.

Below, we compute the axis-ratio of every accepted model sampled by the non-linear search and use this determine the PDF 
of the axis-ratio. When combining the axis-ratio's we weight each value by its `weight`. For Nautilus, a nested sampling 
algorithm, the weight of every sample is different and thus must be included.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the axis-ratio to 
compute its error, we also pass the weight list of the samples.

Note again how because when creating the model above using the input names `lens` and `mass` we access the instance
below using these.
"""
axis_ratio_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model, ignore_assertions=True)

    ell_comps = instance.galaxies.lens.mass.ell_comps

    axis_ratio = al.convert.axis_ratio_from(ell_comps=ell_comps)

    axis_ratio_list.append(axis_ratio)

median_axis_ratio, lower_axis_ratio, upper_axis_ratio = af.marginalize(
    parameter_list=axis_ratio_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"axis_ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")

"""
The calculation above could be computationally expensive, if there are many samples and the derived quantity is
slow to compute.

An alternative approach, which will provide comparable accuracy provided enough draws are used, is to sample 
points randomy from the PDF of the model and use these to compute the derived quantity.

Draws are from the PDF of the model, so the weights of the samples are accounted for and we therefore do not
pass them to the `marginalize` function (it essentially treats all samples as having equal weight).
"""
random_draws = 50

axis_ratio_list = []

for i in range(random_draws):
    instance = samples.draw_randomly_via_pdf()

    ell_comps = instance.galaxies.lens.mass.ell_comps

    axis_ratio = al.convert.axis_ratio_from(ell_comps=ell_comps)

    axis_ratio_list.append(axis_ratio)

median_axis_ratio, lower_axis_ratio, upper_axis_ratio = af.marginalize(
    parameter_list=axis_ratio_list,
    sigma=3.0,
)

print(f"axis_ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")


"""
__Samples Filtering (Advanced)__

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.with_paths(
    [
        ("galaxies", "lens", "mass", "einstein_radius"),
        ("galaxies", "source", "bulge", "sersic_index"),
    ]
)

print(
    "All parameters of the very first sample (containing only the lens mass's einstein radius and "
    "source bulge's sersic index)."
)
print(samples.parameter_lists[0])

print(
    "Maximum Log Likelihood Model Instances (containing only the lens mass's einstein radius and "
    "source bulge's sersic index):\n"
)
print(samples.max_log_likelihood(as_instance=False))

"""
We specified each path as a list of tuples of strings. 

This is how the source code internally stores the path to different components of the model, but it is not 
consistent with the API used to compose a model.

We can alternatively use the following API:
"""
samples = result.samples

samples = samples.with_paths(
    ["galaxies.lens.mass.einstein_radius", "galaxies.source.bulge.sersic_index"]
)

print(
    "All parameters of the very first sample (containing only the lens mass's einstein radius and "
    "source bulge's sersic index)."
)

"""
We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the centres of the mass model to be left with 10 parameters.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("Parameters of first sample")
print(samples.parameter_lists[0])

print(samples.model.total_free_parameters)

samples = samples.without_paths(
    [
        # "galaxies.lens.mass.centre"),
        "galaxies.lens.mass.centre.centre_0",
        # "galaxies.lens.mass.centre.centre_1),
    ]
)

print("Parameters of first sample without the lens mass centre.")
print(samples.parameter_lists[0])

"""
We can keep and remove entire paths of the samples, for example keeping only the parameters of the lens or 
removing all parameters of the source's bulge.
"""
samples = result.samples
samples = samples.with_paths(["galaxies.lens"])
print("Parameters of the first sample of the lens galaxy")
print(samples.parameter_lists[0])

samples = result.samples
samples = samples.without_paths(["galaxies.source.bulge"])
print("Parameters of the first sample without the source's bulge")
print(samples.parameter_lists[0])

"""
Fin.
"""
