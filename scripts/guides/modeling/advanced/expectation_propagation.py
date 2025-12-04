"""
Modeling: Expectation Propagation
=================================

In the `hierarchical` example, we fitted graphical models to a dataset comprising 3 images of strong lenses, which had a
hierarchical parameter, the power-law `slope`. This provides the basis of composing and fitting complex graphical 
models to large datasets.

The challenge is that we will soon hit a ceiling scaling these graphical models up to extremely large datasets. 
One would soon find that the parameter space is too complex to sample, and computational limits would ultimately 
cap how many datasets one could feasible fit.

This example introduces expectation propagation (EP), the solution to this problem, which inspects a factor graph
and partitions the model-fit into many simpler fits of sub-components of the graph to individual datasets. This
overcomes the challenge of model complexity, and mitigates computational restrictions that may occur if one tries to
fit every dataset simultaneously.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/simple__no_lens_light.py`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autofit as af

import numpy as np
from pathlib import Path

"""
__Dataset__

The following steps repeat all the initial steps performed in tutorial 2 and 3:

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/simple__no_lens_light.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "mass_power_law"

dataset_path = Path("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = Path(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        al.Imaging.from_fits(
            data_path=Path(dataset_sample_path, "data.fits"),
            psf_path=Path(dataset_sample_path, "psf.fits"),
            noise_map_path=Path(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__
"""
masked_dataset_list = []

for dataset in dataset_list:
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    masked_dataset_list.append(dataset)

dataset_list = masked_dataset_list

"""
__Model Individual Factors__

We first set up a model for each lens, with an `PowerLawSph` mass and `ExponentialSph` bulge, which we will use to 
fit the hierarchical model.

Note that the `PowerLawSph` mass model has a `slope` parameter, which we will assume is drawn from a shared parent
Gaussian distribution, albeit building this into the model is done later in this script.
"""
model_list = []

for dataset_index in range(total_datasets):

    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
    lens.mass.centre = (0.0, 0.0)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    model_list.append(model)

"""
__Analysis__

For each dataset we now create a corresponding `AnalysisImaging` class, as we are used to doing for `Imaging` data.
"""
analysis_list = []

for dataset in dataset_list:
    analysis = al.AnalysisImaging(dataset=dataset)

    analysis_list.append(analysis)

"""
__Model__

We now compose the hierarchical model that we fit, using the individual model components created above.

This uses the same API as the `hierarchical` example.
"""
hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.TruncatedGaussianPrior(
        mean=2.0, sigma=1.0, lower_limit=0.0, upper_limit=100.0
    ),
    sigma=af.TruncatedGaussianPrior(
        mean=0.5, sigma=0.5, lower_limit=0.0, upper_limit=100.0
    ),
    use_jax=True
)

for model in model_list:
    hierarchical_factor.add_drawn_variable(model.galaxies.lens.mass.slope)

"""
__Paths__
"""
path_prefix = Path("imaging")

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s.

However, unlike the previous tutorials, each `AnalysisFactor` is now assigned its own `search`. This is because the EP 
framework performs a model-fit to each node on the factor graph (e.g. each `AnalysisFactor`). Therefore, each node 
requires its own non-linear search, and in this tutorial we use `dynesty`. For complex graphs consisting of many 
nodes, one could easily use different searches for different nodes on the factor graph.

Each `AnalysisFactor` is also given a `name`, corresponding to the name of the dataset it fits. These names are used
to name the folders containing the results in the output directory.
"""
paths = af.DirectoryPaths(
    path_prefix=path_prefix,
    name="expectation_propagation",
)

search = af.Nautilus(paths=paths, n_live=100)

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):

    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=search, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = af.FactorGraphModel(
    *analysis_factor_list,
    hierarchical_factor,
    use_jax=True
)

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
simply a fit of the `Gaussian` model to the first 1D Gaussian dataset, the model-fit we are used to performing by now.

2) Once the model-fit is complete, inspect the model for parameters that are shared with other nodes on the factor
graph. In this example, the `centre` of the `Gaussian` fitted to the first dataset is global, and therefore connects
to the other nodes on the factor graph (the `AnalysisFactor`'s) of the second and first `Gaussian` datasets.

3) The EP framework now creates a 'message' that is to be passed to the connecting nodes on the factor graph. This
message informs them of the results of the model-fit, so they can update their priors on the `Gaussian`'s centre 
accordingly and, more importantly, update their posterior inference and therefore estimate of the global centre.

For example, the model fitted to the first Gaussian dataset includes the global centre. Therefore, after the model is 
fitted, the EP framework creates a 'message' informing the factor graph about its inference on that Gaussians's centre,
thereby updating our overall inference on this shared parameter. This is termed 'message passing'.

__Cyclic Fitting__

After every `AnalysisFactor` has been fitted (e.g. after each fit to each of the 5 datasets in this example), we have a 
new estimate of the shared parameter `centre`. This updates our priors on the shared parameter `centre`, which needs 
to be reflected in each model-fit we perform on each `AnalysisFactor`. 

The EP framework therefore performs a second iteration of model-fits. It again cycles through each `AnalysisFactor` 
and refits the model, using updated priors on shared parameters like the `centre`. At the end of each fit, we again 
create messages that update our knowledge about other parameters on the graph.

This process is repeated multiple times, until a convergence criteria is met whereby continued cycles are expected to
produce the same estimate of the shared parameter `centre`. 

When we fit the factor graph a `name` is passed, which determines the folder all results of the factor graph are
stored in.
"""
laplace = af.LaplaceOptimiser()

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
)

"""
__Result__

An `info` attribute for the result of a factor graph fitted via EP does not exist yet, its on the to do list!

The result can be seen in the `graph.result` file output to hard-disk.
"""
### print(factor_graph_result.info)##

"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/howtofit/chapter_graphical_models/tutorial_5_expectation_propagation`. 

The following folders and files are worth of note:

 - `graph.info`: this provides an overall summary of the graphical model that is fitted, including every parameter, 
 how parameters are shared across `AnalysisFactor`'s and the priors associated to each individual parameter.

 - The 3 folders titled `gaussian_x1_#__low_snr` correspond to the three `AnalysisFactor`'s and therefore signify 
 repeated non-linear searches that are performed to fit each dataset.

 - Inside each of these folders are `optimization_#` folders, corresponding to each model-fit performed over cycles of
 the EP fit. A careful inspection of the `model.info` files inside each folder reveals how the priors are updated
 over each cycle, whereas the `model.results` file should indicate the improved estimate of model parameters over each
 cycle.

__Results__

The `MeanField` object represent the posterior of the entire factor graph and is used to infer estimates of the 
values and error of each parameter in the graph.
"""
mean_field = factor_graph_result.updated_ep_mean_field.mean_field
print(mean_field)
print()

"""
The object has a `variables` property which lists every variable in the factor graph, which is essentially all of the 
free parameters on the graph.

This includes the parameters specific to each data (E.g. each node on the graph) as well as the shared centre.
"""
print(mean_field.variables)
print()

"""
The variables above use the priors on each parameter as their key. 

Therefore to estimate mean-field quantities of the shared centre, we can simply use the `centre_shared_prior` defined
above.

Each parameter estimate is given by the mean of its value in the `MeanField`. Below, we use the `centred_shared_prior` 
as a key to the `MeanField.mean` dictionary to print the estimated value of the shared centre.
"""
prior = hierarchical_factor.drawn_variables[0]

print(f"Centre Mean Parameter Estimate = {mean_field.mean[prior]}")
print()

"""
If we want the parameter estimate of another parameter in the model, we can use the `model_list` that we composed 
above to pass a parameter prior to the mean field dictionary.
"""
print(
    f"Einstein Radius Dataset 0 Mean = {mean_field.mean[model_list[0].galaxies.lens.mass.einstein_radius]}"
)

"""
The mean-field mean dictionary contains the estimate value of every parameter.
"""
print(f"All Parameter Estimates = {mean_field.mean}")
print()

"""
The mean-field also contains a `variance` dictionary, which has the same keys as the `mean` dictionary above. 

This is the easier way to estimate the error on every parameter, for example that of the shared centre.
"""
print(f"Centre Variance = {mean_field.variance[prior]}")
print()

"""
The standard deviation (or error at one sigma confidence interval) is given by the square root of the variance.
"""
print(f"Centre 1 Sigma = {np.sqrt(mean_field.variance[prior])}")
print()

"""
The mean field object also contains a dictionary of the s.d./variance**0.5.
"""
print(f"Centre SD/sqrt(variance) = {mean_field.scale[prior]}")
print()

