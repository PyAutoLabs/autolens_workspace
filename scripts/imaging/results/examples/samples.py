"""
Results: Samples
================

After fitting strong lens data a search returns a `result` variable. We have used this throughout the
examples scripts to plot the maximum log likelihood tracer and fits.

This `Result` object contains a lot more information which the results tutorials illustrate.

This script describes the non-linear search samples of a model-fit, which for a `DynestyStatic` fit corresponds
to every accepted live point in parameter space.

These are used to compute quantities like the maximum likelihood model, the errors on the parameters and visualization
showing the parameter degeneracies.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The results example `units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.
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

The code below (which we have omitted comments from for brevity) performs a lens model-fit using dynesty. You should
be familiar enough with lens modeling to understand this, if not you should go over the beginner model-fit script again!
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

"""
__Model Composition__

The code below composes the model fitted to the data (the API is described in the `modeling/start_here.py` example).

The way the model is composed below (e.g. that the model is called `cti` and includes a `trap_list` and `ccd`) should 
be noted, as it will be important when inspecting certain results later in this example.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=50,
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
    tracer=result.max_log_likelihood_tracer, grid=mask.derive_grid.all_false_sub_1
)
tracer_plotter.subplot_tracer()
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Results tutorials 2 and 3 will expand on the `max_log_likelihood_tracer` and `max_log_likelihood_fit`, showing how 
they can be used to inspect many aspects of a model.

__Samples__

The result's `Samples` object contains the complete set of non-linear search dynesty samples, where each sample 
corresponds to a set of a model parameters that were evaluated and accepted. 

This also includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the nested sampling algorithm Dynesty, so the `Samples` object returned is a `SamplesNest` object.
"""
samples = result.samples

print("Nest Samples: \n")
print(samples)

"""
__Parameters__

The `Samples` class contains all the parameter samples, which is a list of lists where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
print("All parameters of the very first sample")
print(samples.parameter_lists[0])
print("The fourth parameter of the tenth sample")
print(samples.parameter_lists[9][3])

"""
__Figures of Merit__

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every sample, where:

 - The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
   normalization).

 - The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log
   posterior value.

 - The log posterior is log_likelihood + log_prior.

 - The weight gives information on how samples should be combined to estimate the posterior. The weight values 
   depend on the sampler used. For example for an MCMC search they will all be 1`s whereas for the nested sampling
   method used in this example they are weighted as a combination of the log likelihood value and prior..
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Instances__

The `Samples` contains many results which are returned as an instance of the model, using the Python class structure
of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.
"""
max_lh_instance = samples.max_log_likelihood()
print("Maximum Log Likelihood Model Instance: \n")
print(max_lh_instance, "\n")

"""
A model instance contains all the model components of our fit, for example the list of galaxies we specified during 
model composition.
"""
print(max_lh_instance.galaxies)

"""
These galaxies will be named according to the model fitted by the search (in this case, `lens` and `source`).
"""
print(max_lh_instance.galaxies.lens)
print(max_lh_instance.galaxies.source)

"""
Their `LightProfile`'s and `MassProfile`'s are also named according to the search.
"""
print(max_lh_instance.galaxies.lens.mass)

"""
We can use this list of galaxies to create the maximum log likelihood `Tracer`, which, funnily enough, 
is the property of the result we've used up to now!

Using this tracer will be expanded upon in the next results tutorial.

(If we had the `Imaging` available we could easily use this to create the maximum log likelihood `FitImaging`).
"""
max_lh_tracer = al.Tracer.from_galaxies(galaxies=max_lh_instance.galaxies)

print(max_lh_tracer)
print(mask.derive_grid.all_false_sub_1)

tracer_plotter = aplt.TracerPlotter(
    tracer=max_lh_tracer, grid=mask.derive_grid.all_false_sub_1
)
tracer_plotter.subplot_tracer()

"""
__Vectors__

All results can alternatively be returned as a 1D vector of values, by passing `as_instance=False`:
"""
max_lh_vector = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_vector, "\n\n")

"""
__Labels__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:

 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models illustrated in other tutorials their utility will become clear.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
__Posterior / PDF__

PDF stands for "Probability Density Function" and it quantifies the PDF stands for "Probability Density Function" and it quantifies the probability of each model sampled. It 
therefore enables error estimation via a process called marginalization.

We can access the `median pdf` model, which is the model computed by marginalizing over the samples of every 
parameter in 1D and taking the median of this PDF.
"""
median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance, "\n")
print(median_pdf_instance.galaxies.source.bulge)
print()

median_pdf_vector = samples.median_pdf(as_instance=False)

print("Median PDF Model Parameter Lists: \n")
print(median_pdf_vector, "\n")


"""
__Errors__

We can compute the model parameters at a given sigma value (e.g. at 3.0 sigma limits).

These parameter values and error estimates do not fully account for covariance between model parameters, which is
explained in the "Derived Errors" section below.

The `uv3` below signifies this is an upper value at 3 sigma confidence, with `lv3` indicating a the lower value.
"""
uv3_instance = samples.values_at_upper_sigma(sigma=3.0)
lv3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(uv3_instance, "\n")
print(lv3_instance, "\n")

"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

The `ue3` below signifies the upper error at 3 sigma. 
"""
ue3_instance = samples.errors_at_upper_sigma(sigma=3.0)
le3_instance = samples.errors_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance, "\n")
print(le3_instance, "\n")

"""
__Search Plots__

The Probability Density Functions of the results can be plotted using Dynesty's in-built visualization tools, 
which are wrapped via the `DynestyPlotter` object.
"""
search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
__Sample Instance__

A dynesty search retains every model that is accepted during the model-fit.

We can create an instance of any lens model -- below we create an instance of the 100th last accepted model, and we
can compare its parameters to the maximum log likelihood model.
"""
instance = samples.from_sample_index(sample_index=-10)

print(max_lh_instance.galaxies.lens.mass)
print(instance.galaxies.lens.mass)

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

Nested sampling algorithms like dynesty also estimate the Bayesian evidence (estimated via the nested sampling 
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
__Latex__

If you are writing results up in a paper, you can use PyAutoFit's inbuilt latex tools to create latex 
table code which you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

The superscripts of each parameter's latex string are loaded from the config file `notation/label.yaml`. Editing this
config file provides high levels of customization for each parameter appears in the latex table. 

This is especially useful if your lens model uses the same model components with the same parameter, which 
therefore need to be distinguished via superscripts.
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
__Wrap Up__

This tutorial illustrated how to analyse the non-linear samples of a model-fit. We saw that the API used for model 
composition produced the instances of the results after fitting (for example, how we accessed the galaxy model 
as `instance.galaxies.lens.mass`).

The remainder of this script provides advanced samples calculations which:

 - Calculate the most probable value with errors of a derived quantity by computing its PDF. For example, computing 
   the axis-ratio of the model with errors (the axis-ratio is not directly accessible from the model).
 
 - Filtering `Samples` objects to remove certain parameters or components from the model. This can help with,
   for example, creating LaTex tables of specific parts of a model.

__Derived Errors (Advanced)__

Computing the errors of a quantity like the `einstein_radius` is simple, because it is sampled by the non-linear 
search. Errors are accessible using the `Samples` object's `errors_from` methods, which marginalize over the 
parameters via the 1D Probability Density Function (PDF).

Computing errors on derived quantitys is more tricky, because it is not sampled directly by the non-linear search. 
For example, what if we want the error on the axis-ratio of the mass model? In order to do this we need to create the 
PDF of that derived quantity, which we can then marginalize over using the same function we use to marginalize model 
parameters.

Below, we compute the axis-ratio of every accepted model sampled by the non-linear search and use this determine the PDF 
of the axis-ratio. When combining the axis-ratio's we weight each value by its `weight`. For Dynesty, a nested sampling 
algorithm, the weight of every sample is different and thus must be included.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the axis-ratio to 
compute its error, we also pass the weight list of the samples.

Note again how because when creating the model above using the input names `lens` and `mass` we access the instance
below using these.
"""
axis_ratio_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model)

    ell_comps = instance.galaxies.lens.mass.ell_comps

    axis_ratio = al.convert.axis_ratio_from(ell_comps=ell_comps)

    axis_ratio_list.append(axis_ratio)

median_axis_ratio, upper_axis_ratio, lower_axis_ratio = af.marginalize(
    parameter_list=axis_ratio_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"axis_ratio = {median_axis_ratio} ({upper_axis_ratio} {lower_axis_ratio}")


"""
__Samples Filtering (Advanced)__

The samples object has the results for all model parameter. It can be filtered to contain the results of specific 
parameters of interest.

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
Above, we specified each path as a list of tuples of strings. 

This is how the PyAutoFit source code stores the path to different components of the model, but it is not in-line 
with the PyAutoLens API used to compose a model.

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
