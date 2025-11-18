"""
Tutorial 6: Science Case
========================

This tutorial shows using a graphical model and EP for a realistic science case, fitting a sample of time-delay lensed
quasars using a graphical model, where time delays allow us include the Cosmological parameter the Hubble constant H0
as a shared free parameter in an graphical model.

In this example we fit via a graphical model and Expectation Propagation (EP).

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/advanced/hubble_constant_time_delays.py`.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
from pathlib import Path
import autolens as al
import autofit as af

"""
__Initialization__

The following steps repeat all the initial steps performed in tutorial 2 and 3:

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/advanced/mass_power_law.py`. 
"""
dataset_label = "samples"
dataset_type = "point_source"
dataset_sample_name = "hubble_constant_time_delays"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_sample_name

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = dataset_path / f"dataset_{dataset_index}"

    dataset = al.from_json(
        file_path=dataset_sample_path / "point_dataset_with_time_delays.json",
    )

    dataset_list.append(dataset)

"""
__Point Solver__

We set up the `PointSolver`, which is used to compute the multiple images of the point source in the image-plane.

There are no special settings or inputs for the fitting of time_delays, therefore the `PointSolver` is set up in the same way
as in the `modeling/start_here.ipynb` notebook.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1, xp=jnp
)

"""
__Paths__
"""
path_prefix = Path("point_source") / "hierarchical"

"""
__Model__

We compose our model using `Model` objects, which represent the lenses we fit to our data.

This graphical model creates a non-linear parameter space that has parameters for every lens mass and source galaxy point
source in our sample. In this example, there are 3 lenses each with their own model, therefore:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 
 - The source galaxy's light is a point `Point` [2 parameters].

 - There is a single cosmological shared free parameter, `H0` [1 parameter]

 - There are 3 strong lenses in our graphical model [(3 x 7) + 1 = 22 parameters]. 

The overall dimensionality of each parameter space fitted separately via EP is therefore N=7.

In total, the graph has N = 3 x 7 + 1 = 2 free parameters, albeit EP knows the `H0` is shared and fits it 
using EP.
"""
cosmology = af.Model(al.cosmo.FlatwCDMWrap)

cosmology.H0 = af.UniformPrior(lower_limit=0.0, upper_limit=150.0)

model_list = []

for model_index in range(total_datasets):

    # Lens:

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = 0.0
    mass.centre.centre_1 = 0.0

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    # Source:

    point_0 = af.Model(al.ps.Point)

    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

    # Overall Lens Model:

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        cosmology=cosmology,
    )

    model_list.append(model)

"""
__Analysis__
"""
analysis_list = []

for dataset in dataset_list:

    analysis = al.AnalysisPoint(dataset=dataset, solver=solver)

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
search = af.Nautilus(
    path_prefix=Path("point_source") / "hierarchical",
    name="tutorial_6_science_case_graphical",
    n_live=150,
)

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
We again combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
The factor graph model `info` attribute shows the complex model we are fitting, including both cosmological parameters.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now use the search to fit factor graph, using its `global_prior_model` property.
"""
result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Expectation Propagation__

We now perform the fit using EP as we did in tutorial 5.
"""
paths = af.DirectoryPaths(
    name=Path("point_source") / "hierarchical" / "tutorial_6_science_case_ep"
)

search = af.Nautilus(
    path_prefix=Path("point_source") / "hierarchical",
    name="tutorial_6_science_case_ep",
    n_live=150,
)

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):
    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=search, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

laplace = af.LaplaceOptimiser()

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
)

"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/graphical/imaging/tutorial_6_science_case`. 
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
