"""
Tutorial 6: Science Case
========================

This tutorial shows a realistic science case.

We have a dataset containing 10 double Einstein ring lenses, which allow one to measure certain Cosmological
parameters.

In this example we include the Cosmological parameter Omega_m as a shared free parameter in an graphical model fit
via Expectation Propagation (EP).

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/double_einstein_ring.py`.
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
__Initialization__

The following steps repeat all the initial steps performed in tutorial 2 and 3:

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/simulators/imaging/samples/advanced/mass_power_law.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "double_einstein_ring"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 10

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

 - The first lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 
 - The second lens / first source galaxy's light is a linear parametric `ExponentialSph` and its mass 
 a `IsothermalSph` [6 parameters].
 
 - The second source galaxy's light is a linear parametric `ExponentialSph` [3 parameters].

 - There is a single cosmological shared free parameter, Omage_m [1 parameter]

 - There are ten strong lenses in our graphical model [(10 x 16) + 1 = 161 parameters]. 

The overall dimensionality of each parameter space fitted separately via EP is therefore N=17.

In total, the graph has N = 10 x 16 + 1 = 161 free parameters, albeit EP knows the `Omage_k` is shared and fits it 
using EP.

__CHEATING__

Initializing a double Einstein ring lens model is extremely difficult, due to the complexity of parameter space. It is
common to infer local maxima, which this script typically does if default priors on every model parameter are 
assumed.

To ensure we infer the correct model, we therefore cheat and overwrite all of the priors of the model parameters to 
start centred on their true values.

To model a double Einstein ring system without cheating (which is the only feasible strategy on real data), it is 
advised that **PyAutoLens**'s advanced feature of non-linear search chaining is used. The 
scripts `imaging/chaining/double_einstein_ring.py`  and `imaging/pipelines/double_einstein_ring.py` describe how to 
do this.
"""
shared_cosmology_parameter = af.GaussianPrior(
    mean=0.3, sigma=0.3, lower_limit=0.0, upper_limit=1.0
)

model_list = []

for model_index in range(total_datasets):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
    source_0 = af.Model(
        al.Galaxy,
        redshift=1.0,
        mass=al.mp.IsothermalSph,
        bulge=al.lp_linear.ExponentialCoreSph,
    )
    source_1 = af.Model(al.Galaxy, redshift=2.0, bulge=al.lp_linear.ExponentialCoreSph)

    lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    lens.mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.052, sigma=0.1)
    lens.mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    lens.mass.einstein_radius = af.GaussianPrior(mean=1.5, sigma=0.2)

    source_0.mass.centre_0 = af.GaussianPrior(mean=-0.15, sigma=0.2)
    source_0.mass.centre_1 = af.GaussianPrior(mean=-0.15, sigma=0.2)
    source_0.mass.einstein_radius = af.GaussianPrior(mean=0.4, sigma=0.1)
    source_0.bulge.centre_0 = af.GaussianPrior(mean=-0.15, sigma=0.2)
    source_0.bulge.centre_1 = af.GaussianPrior(mean=-0.15, sigma=0.2)
    source_0.bulge.intensity = af.GaussianPrior(mean=1.2, sigma=0.5)
    source_0.bulge.effective_radius = af.GaussianPrior(mean=0.1, sigma=0.1)

    source_1.bulge.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
    source_1.bulge.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.2)
    source_1.bulge.intensity = af.GaussianPrior(mean=0.6, sigma=0.3)
    source_1.bulge.effective_radius = af.GaussianPrior(mean=0.07, sigma=0.07)

    cosmology = af.Model(al.cosmo.FlatLambdaCDMWrap)
    cosmology.Om0 = af.GaussianPrior(mean=0.3, sigma=0.1)

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1),
        cosmology=cosmology,
    )

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
    name="tutorial_6_science_case",
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
The factor graph model `info` attribute shows the complex model we are fitting, including both cosmological parameters.
"""
print(factor_graph.global_prior_model.info)

"""
__Expectation Propagation__

We perform the fit using EP as we did in tutorial 5.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(name=path.join(path_prefix, "tutorial_6_science_case"))

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
