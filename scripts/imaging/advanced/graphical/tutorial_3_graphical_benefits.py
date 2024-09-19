"""
Tutorial 3: Graphical Benefits
==============================

In the previous tutorials, we fitted a dataset containing 3 lenses which had a shared `slope` value.

We used different approaches to estimate the shared `slope`, for example a simple approach of fitting each
dataset one-by-one and estimating the slope via a weighted average or posterior multiplication and a more
complicated approach using a graphical model.

The estimates were consistent with one another, making it hard to justify the use of the more complicated graphical
model. However, the model fitted in the previous tutorial was extremely simple, and by making it slightly more complex
in this tutorial we will be able to show the benefits of using the graphical modeling approach.

__The Model__

The more complex datasets and model fitted in this tutorial is an extension of those fitted in the previous tutorial.

Previously, the slope of each lens galaxy mass distribution was a power-law controlled but ojnly a slope.

In this tutorial we fit a broken power-law with two parameters controlling the slope, which are shared across the
dataset.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/mass_bpl.py`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

For each lens dataset in our sample we set up the correct path and load it by iterating over a for loop. 

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/mass_bpl.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "mass_bpl"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        al.Imaging.from_fits(
            data_path=path.join(dataset_sample_path, "data.fits"),
            psf_path=path.join(dataset_sample_path, "psf.fits"),
            noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__

We now mask each lens in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every lens in the dataset is appropriate.
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    masked_imaging_list.append(dataset.apply_mask(mask=mask))

"""
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = path.join("imaging", "hierarchical")

"""
__Model (one-by-one)__

We are first going to fit each dataset one by one.

We therefore fit a model where

 - The lens galaxy's total mass distribution is an `SphBrokenPowerLaw` with multiple parameters fixed to their true 
 values [3 parameter].
 
 - The source galaxy's light is a linear parametric `ExponentialSph` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

To make graphical model fits run fast, the model above is simple compared to a lot of models fitted throughout the 
workspace (for example, both galaxies are spherical).

If you are not familiar with the broken power-law, it is an extension of the power-law where two parameters control
the slope instead of one. The broken power-law reduces to the power-law when `inner_slope=1.0` and `outer_slope=1.0`.
For the simulated data fitted in this tutorial, all len mass models assume `inner_slope=1.5` and `outer_slope=0.5`.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawBrokenSph)
lens.mass.centre = (0.0, 0.0)
lens.mass.break_radius = 0.01

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class. 
"""
analysis_list = []

for dataset_index, masked_dataset in enumerate(masked_imaging_list):
    analysis = al.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

"""
__Model Fits (one-by-one)__

For each dataset we now create a non-linear search, analysis and perform the model-fit using this model.

The `Result` is stored in the list `result_list` and they are output to a unique folder named using the `dataset_index`..
"""
result_list = []

for dataset_index, analysis in enumerate(analysis_list):
    dataset_name_with_index = f"dataset_{dataset_index}"
    path_prefix_with_index = path.join(path_prefix, dataset_name_with_index)

    search = af.Nautilus(
        path_prefix=path_prefix_with_index, name=dataset_name_with_index, n_live=150
    )

    result = search.fit(model=model, analysis=analysis)
    result_list.append(result)

"""
__Slope Estimates (Weighted Average)__

We can now compute the slope estimate of the mass profiles, including their errors, from the individual model fits
performed above.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf() for samps in samples_list]
mp_inner_slope = [instance.galaxies.lens.mass.inner_slope for instance in mp_instances]

ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_inner_slope = [
    instance.galaxies.lens.mass.inner_slope for instance in ue1_instances
]
le1_inner_slope = [
    instance.galaxies.lens.mass.inner_slope for instance in le1_instances
]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_inner_slope, le1_inner_slope)]

values = np.asarray(mp_inner_slope)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

weighted_inner_slope = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error_inner_slope = 1.0 / np.sqrt(weight_averaged)


mp_instances = [samps.median_pdf() for samps in samples_list]
mp_outer_slope = [instance.galaxies.lens.mass.outer_slope for instance in mp_instances]

ue1_instances = [samp.values_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.values_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_outer_slope = [
    instance.galaxies.lens.mass.outer_slope for instance in ue1_instances
]
le1_outer_slope = [
    instance.galaxies.lens.mass.outer_slope for instance in le1_instances
]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_outer_slope, le1_outer_slope)]

values = np.asarray(mp_outer_slope)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

weighted_outer_slope = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error_outer_slope = 1.0 / np.sqrt(weight_averaged)


print(
    f"Weighted Average Inner Slope Estimate = {weighted_inner_slope} ({weighted_error_inner_slope}) [1.0 sigma confidence intervals]"
)
print(
    f"Weighted Average Outer Slope Estimate = {weighted_outer_slope} ({weighted_error_outer_slope}) [1.0 sigma confidence intervals]"
)

"""
The estimate of the slopes are not accurate, with both estimates well offset from the input values 
of 1.5 and 0.5

We will next show that the graphical model offers a notable improvement, but first lets consider why this
approach is suboptimal.

The most important difference between this model and the model fitted in the previous tutorial is that there are now
two shared parameters we are trying to estimate, *and they are degenerate with one another*.

We can see this by inspecting the probability distribution function (PDF) of the fit, placing particular focus on the 
2D degeneracy between the inner slope and outer slope of the lens mass model.
"""
plotter = aplt.NestPlotter(samples=result_list[0].samples)
plotter.corner_anesthetic()

"""
The problem is that the simple approach of taking a weighted average does not capture the curved banana-like shape
of the PDF between the two slope. This leads to significant error over estimation and biased inferences on the 
estimates.

__Discussion__

Let us now consider other downsides of fitting each dataset one-by-one, from a more statistical perspective. We 
will contrast these to the graphical model later in the tutorial.

1) By fitting each dataset one-by-one this means that each model-fit fails to fully exploit the information we know 
about the global model. We *know* that there are only two single shared values of `slope` across the full dataset 
that we want to estimate. However, each individual fit has its own `slope` value which is able to assume 
different values than the `slope` values used to fit the other datasets. This means that the large degeneracies 
between the two slope emerge for each model-fit.

By not fitting our model as a global model, we do not maximize the amount of information that we can extract from the 
dataset as a whole. If a model fits dataset 1 particularly bad, this *should* be reflected in how we interpret how 
well the model fits datasets 2 and 3. Our non-linear search should have a global view of how well the model fits the 
whole dataset. This is the *crucial aspect of fitting each dataset individually that we miss*, and what a graphical 
model addresses.

2) When we combined the result to estimate the global `slope` value via a weighted average, we marginalized over 
the samples in 1D. As showed above, when there are strong degeneracies between models parameters the information on 
the covariance between these parameters is lost when computing the global `slope`. This increases the inferred 
uncertainties. A graphical model performs no such 1D marginalization and therefore fully samples the
parameter covariances.

3) In Bayesian inference it is important that we define priors on all of the model parameters. By estimating the 
global `slope` after the model-fits are completed it is unclear what prior the global `slope` a
ctually has! We actually defined the prior five times -- once for each fit -- which is not a well defined prior.

In a graphical model the prior is clearly defined.

What would have happened if we had estimate the shared slope via 2D posterior multiplication using a KDE? We
will discuss this at the end of the tutorial after fitting a graphical model.

__Model (Graphical)__

We now compose a graphical model and fit it.

Our model now consists of a lens mass model with a broken power-law, where the inner slope and outer slope have a
`slope_shared_prior` variable, such that the same `inner_slope` and `outer_slope` parameters are used for the mass
model of all lenses in all datasets. 
"""
inner_slope_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
outer_slope_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

"""
We now set up a list of `Model`'s, each of which contain a broken power law mass model.

All of these `Model`'s use the `slope_shared_prior`'s above. This means all model-components use the same value 
of `inner_slope` and `outer_slope`.

For a fit to three datasets this produces a parameter space with dimensionality N=15 (1 parameter per mass model, 
4 parameters per source galaxy and 2 shared `inner_slope` and `outer_slope` parameters).
"""
model_list = []


for model_index in range(total_datasets):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawBrokenSph)
    lens.mass.centre = (0.0, 0.0)

    # This makes every Galaxy share the same `inner_slope` and `outer_slope`.
    lens.mass.inner_slope = inner_slope_shared_prior
    lens.mass.outer_slope = outer_slope_shared_prior

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    model_list.append(model)

"""
__Analysis Factors__

We again create the graphical model using `AnalysisFactor` objects.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

The analysis factors are then used to create the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
The factor graph model can again be printed via the `info` attribute, which shows that there are two shared
parameters across the datasets.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and use it to the fit the factor graph, again using its `global_prior_model` 
property.
"""
search = af.Nautilus(
    path_prefix=path_prefix,
    name="tutorial_3_graphical_benefits_2",
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same structure of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
We can now inspect the inferred `inner_slope` and `outer_slope` values and compare this to the values estimated above 
via a weighted average.  

(The errors of the weighted average is what was estimated for a run on my PC, yours may be slightly different!)
"""
inner_slope = result.samples.median_pdf()[0].galaxies.lens.mass.inner_slope

u1_error_0 = result.samples.values_at_upper_sigma(sigma=1.0)[
    0
].galaxies.lens.mass.inner_slope
l1_error_0 = result.samples.values_at_lower_sigma(sigma=1.0)[
    0
].galaxies.lens.mass.inner_slope

u3_error_0 = result.samples.values_at_upper_sigma(sigma=3.0)[
    0
].galaxies.lens.mass.inner_slope
l3_error_0 = result.samples.values_at_lower_sigma(sigma=3.0)[
    0
].galaxies.lens.mass.inner_slope

outer_slope = result.samples.median_pdf()[0].galaxies.lens.mass.outer_slope

u1_error_1 = result.samples.values_at_upper_sigma(sigma=1.0)[
    0
].galaxies.lens.mass.outer_slope
l1_error_1 = result.samples.values_at_lower_sigma(sigma=1.0)[
    0
].galaxies.lens.mass.outer_slope

u3_error_1 = result.samples.values_at_upper_sigma(sigma=3.0)[
    0
].galaxies.lens.mass.outer_slope
l3_error_1 = result.samples.values_at_lower_sigma(sigma=3.0)[
    0
].galaxies.lens.mass.outer_slope


print(
    f"Weighted Average Inner Slope  Estimate = 1.8793105272514588 (0.1219903793654069) [1.0 sigma confidence intervals]\n"
)
print(
    f"Weighted Average Outer Slope Estimate = 1.3589940186100282 (0.08932284400100543) [1.0 sigma confidence intervals]\n"
)

print(
    f"Inferred value of the inner slope via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{inner_slope} ({l1_error_0} {u1_error_0}) ({u1_error_0 - l1_error_0}) [1.0 sigma confidence intervals]"
)
print(
    f"{inner_slope} ({l3_error_0} {u3_error_0}) ({u3_error_0 - l3_error_0}) [3.0 sigma confidence intervals]"
)

print(
    f"Inferred value of the outer slope via a graphical fit to {total_datasets} datasets: \n"
)
print(
    f"{outer_slope} ({l1_error_1} {u1_error_1}) ({u1_error_1 - l1_error_1}) [1.0 sigma confidence intervals]"
)
print(
    f"{outer_slope} ({l3_error_1} {u3_error_1}) ({u3_error_1 - l3_error_1}) [3.0 sigma confidence intervals]"
)

"""
As expected, using a graphical model allows us to infer a more precise and accurate model.

You may already have an idea of why this is, but lets go over it in detail:

__Discussion__

Unlike a fit to each dataset one-by-one, the graphical model:

1) Infers a PDF on the global slope that fully accounts for the degeneracies between the models fitted to 
different datasets. This reduces significantly the large 2D degeneracies between the two slope we saw when 
inspecting the PDFs of each individual fit.

2) Fully exploits the information we know about the global model, for example that the slope of every lens
in every dataset is aligned. Now, the fit of the lens in dataset 1 informs the fits in datasets 2 and 3, and visa 
versa.

3) Has a well defined prior on the global slope, instead of independent priors on the slope of each 
dataset.

__Posterior Multiplication__

What if we had combined the results of the individual model fits using 2D posterior multiplication via a KDE?

This would produce an inaccurate estimate of the error, because each posterior contains the prior on the slope 
multiple times which given the properties of this model should not be repeated.

However, it is possible to convert each posterior to a likelihood (by dividing by its prior), combining these
likelihoods to form a joint likelihood via 2D KDE multiplication and then insert just one prior back (agian using a 2D
KDE) at the end to get a posterior which does not have repeated priors. 

This posterior, in theory, should be equivalent to the graphical model, giving the same accurate estimates of the
slope with precise errors. The process extracts the same information, fully accounting for the 2D structure 
of the PDF between the two slope for each fit.

However, in practise, this will likely not work that well. Every time we use a KDE to represent and multiply a 
posterior, we make an approximation which will impact our inferred errors. The removal of the prior before combining 
the likelihood and reinserting it after also introduces approximations, especially because the fit performed by the 
non-linear search is informed by the prior. 

Crucially, whilst posterior multiplication maybe sort-of-works-ok in two dimensions, for models with many more 
dimensions and degeneracies between parameters that are in 3D, 4D of more dimensions it simply does not work.

In contrast, a graphical model fully samples all of the information a large dataset contains about the model, without
making an approximations. In this sense, irrespective of how complex the model gets, it will fully extract the 
information contained in the dataset.

__Wrap Up__

In this tutorial, we demonstrated the strengths of a graphical model over fitting each dataset one-by-one. 

We argued that irrespective of how one may try to combine the results of many individual fits, the approximations that 
are made will always lead to a suboptimal estimation of the model parameters and fail to fully extract all information
from the dataset. 

Furthermore, we argued that for high dimensional complex models a graphical model is the only way to fully extract
all of the information contained in the dataset.

In the next tutorial, we will consider a natural extension of a graphical model called a hierarchical model.
"""
