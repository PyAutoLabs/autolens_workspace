"""
Tutorial 5: Expectation Propagation
===================================

In the previous tutorial, we fitted graphical models to a dataset comprising 3 images of strong lenses, which had a
shared and global value of `slope` for the lens's mass distribution, or assumed their slopes were hierarchically drawn
from a parent Gaussian distribution. This provides the basis of composing and fitting complex graphical models to large
datasets.

We concluded by discussing that one would soon hit a ceiling scaling these graphical models up to extremely large
datasets. One would soon find that the parameter space is too complex to sample, and computational limits would
ultimately cap how many datasets one could feasible fit.

This tutorial introduces expectation propagation (EP), the solution to this problem, which inspects a factor graph
and partitions the model-fit into many simpler fits of sub-components of the graph to individual datasets. This
overcomes the challenge of model complexity, and mitigates computational restrictions that may occur if one tries to
fit every dataset simultaneously.

This tutorial fits a global model with a shared parameter and does not use a hierarchical model. Using a
hierarchical model uses the same API introduced in tutorial 3, whereby a `HierarchicalFactor` is created
and passed to the `FactorGraphModel`.

The model fitted in this tutorial is the simpler model fitted in tutorials 1 & 2, where the weighted average
proivided an accurate estimate of the shared parameter. We fit the same simple model here to illustrate EP, and will
fit a more challenging model that is only possible because of EP in the next tutorial.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/simple__no_lens_light.py`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autofit as af
from os import path

"""
__Dataset__

The following steps repeat all the initial steps performed in tutorial 2 and 3:

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/simulators/imaging/samples/simple__no_lens_light.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple__no_lens_light__mass_sis"

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
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    masked_imaging_list.append(dataset.apply_mask(mask=mask))

"""
__Paths__
"""
path_prefix = path.join("imaging", "hierarchical")

"""
__Model__

We compose our model using `Model` objects, which represent the lenses we fit to our data.

This graphical model creates a non-linear parameter space that has parameters for every lens and source galaxy in our 
sample. In this example, there are 3 lenses each with their own model, therefore:

 - Each lens galaxy's total mass distribution is an `PowerLawSph` with its centre fixed to its true value of 
 (0.0, 0.0) [2 parameters].
 
 - Each source galaxy's light is a linear parametric `ExponentialSph` [3 parameters].

 - There are three lenses in our graphical model [3 x 1 = 3 parameters]. 

 - There are three source in our graphical model [3 x 4 = 12 parameters]. 

The overall dimensionality of each parameter space fitted separately via EP is therefore N=6.

In total, the graph has N = 3 x 6 = 18 free parameters, albeit EP knows the `slope` is shared and fits it in the special
way described below to account for this.
"""
slope_shared_prior = af.GaussianPrior(mean=2.0, sigma=1.0)

model_list = []

for model_index in range(total_datasets):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
    lens.mass.centre = (0.0, 0.0)
    lens.mass.einstein_radius = af.GaussianPrior(mean=1.8, sigma=0.3, lower_limit=0.0)

    # This slope is shared across all lens galaxies
    lens.mass.slope = slope_shared_prior

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCore)

    source.bulge.ell_comps.ellipitcal_comps_0 = 0.0
    source.bulge.ell_comps.ellipitcal_comps_1 = 0.0

    source.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source.bulge.effective_radius = af.GaussianPrior(
        mean=1.0, sigma=1.0, lower_limit=0.01, upper_limit=3.0
    )

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    model_list.append(model)

"""
__Analysis__
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = al.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s, just like we did in the
previous tutorial.

However, unlike the previous tutorial, each `AnalysisFactor` is now assigned its own `search`. This is because the EP 
framework performs a model-fit to each node on the factor graph (e.g. each `AnalysisFactor`). Therefore, each node 
requires its own non-linear search. 

For complex graphs consisting of many  nodes, one could easily use different searches for different nodes on the factor 
graph.
"""
Nautilus = af.Nautilus(
    path_prefix=path.join("imaging", "hierarchical"),
    name="tutorial_5_expectation_propagation",
    n_live=150,
)

analysis_factor_list = []
dataset_index = 0

for model, analysis in zip(model_list, analysis_list):
    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=Nautilus, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
We again combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
The factor graph model `info` attribute shows the model which we fit via expectaton propagation (note that we do
not use `global_prior_model` below when performing the fit).
"""
print(factor_graph.global_prior_model.info)

"""
__Expectation Propagation__

In the previous tutorials, we used the `global_prior_model` of the `factor_graph` to fit the global model. In this 
tutorial, we instead fit the `factor_graph` using the EP framework, which fits the graphical model composed in this 
tutorial as follows:

1) Go to the first node on the factor graph (e.g. `analysis_factor_list[0]`) and fit its model to its dataset. This is 
simply a fit of the first lens model to the first lens data dataset, the model-fit we are used to performing by now.

2) Once the model-fit is complete, inspect the model for parameters that are shared with other nodes on the factor
graph. In this example, the `slope` of the mass model fitted to the first dataset is global, and therefore connects
to two other nodes on the factor graph (the `AnalysisFactor`'s) of the second and first lens datasets.

3) The EP framework now creates a 'message' that is to be passed to the connecting nodes on the factor graph. This
message informs them of the results of the model-fit, so they can update their priors on the slope accordingly and, 
more importantly, update their posterior inference and therefore estimate of the global slope.

For example, the model fitted to the first lens dataset includes the global slope. Therefore, after the model is 
fitted, the EP framework creates a 'message' informing the factor graph about its inference on that lens model's slope,
thereby updating our overall inference on this shared parameter. This is termed 'message passing'.

__Cyclic Fitting__

After every `AnalysisFactor` has been fitted (e.g. all 3 datasets in this example), we have a new estimate of the 
shared parameter `slope`. This updates our priors on the shared parameter `slope`, which needs to be reflected in 
each model-fit we perform on each `AnalysisFactor`. 

The EP framework therefore performs a second iteration of model-fits. It again cycles through each `AnalysisFactor` 
and refits the model, using updated priors on shared parameters like the `slope`. At the end of each fit, we again 
create messages that update our knowledge about other parameters on the graph.

This process is repeated multiple times, until a convergence criteria is met whereby continued cycles are expected to
produce the same estimate of the shared parameter `slope`. 

When we fit the factor graph a `name` is passed, which determines the folder all results of the factor graph are
stored in.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join(path_prefix, "tutorial_5_expectation_propagation_2")
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
)

"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/graphical/imaging/tutorial_5_expectation_propagation`. 

The following folders and files are worth of note:

 - `graph.info`: this provides an overall summary of the graphical model that is fitted, including every parameter, 
 how parameters are shared across `AnalysisFactor`'s and the priors associated to each individual parameter.
 
 - The 3 folders titled `gaussian_x1_#__low_snr` correspond to the three `AnalysisFactor`'s and therefore signify 
 repeated non-linear searches that are performed to fit each dataset.
 
 - Inside each of these folders are `optimization_#` folders, corresponding to each model-fit performed over cycles of
 the EP fit. A careful inspection of the `model.info` files inside each folder reveals how the priors are updated
 over each cycle, whereas the `model.results` file should indicate the improved estimate of model parameters over each
 cycle.
"""

print(factor_graph_result)

print(factor_graph_result.updated_ep_mean_field.mean_field)

"""
__Output__

The MeanField object representing the posterior.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field)
print()

print(factor_graph_result.updated_ep_mean_field.mean_field.variables)
print()

"""
The logpdf of the posterior at the point specified by the dictionary values
"""
# factor_graph_result.updated_ep_mean_field.mean_field(values=None)
print()

"""
A dictionary of the mean with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.mean)
print()

"""
A dictionary of the variance with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.variance)
print()

"""
A dictionary of the s.d./variance**0.5 with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.scale)
print()

"""
self.updated_ep_mean_field.mean_field[v: Variable] gives the Message/approximation of the posterior for an 
individual variable of the model.
"""
# factor_graph_result.updated_ep_mean_field.mean_field["help"]

"""
Finish.
"""
