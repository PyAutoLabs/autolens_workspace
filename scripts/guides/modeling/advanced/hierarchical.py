"""
Modeling: Hierarchical
======================

A hierarchical model assumes that certain model parameters are drawn from a **shared parent distribution**
(e.g. a Gaussian). When we fit such a model, the parameters of this parent distribution (such as its `mean` and
`sigma`) become explicit free parameters in the inference. Scientifically, these parent-distribution parameters
are often of greatest interest because they describe **population-level trends**, rather than the properties of
individual lenses.

In this example, we fit a hierarchical model to a sample of three strong gravitational lenses. We assume that
the **power-law slope** of each lens’s mass distribution is drawn from a shared Gaussian distribution. This is
well motivated: observational studies find that the slopes of early-type lens galaxies are well approximated by
a Gaussian with mean ≈ 2.06 and sigma ≈ 0.20.

To perform this fit, we use a graphical model (see `guides/modeling/advanced/graphical`). The model-composition
API makes it straightforward to fit multiple datasets simultaneously while linking parameters via a shared
parent distribution.

Note that hierarchical models **do not have to be fit through graphical models**—the same API can be applied to
single-object problems where multiple components of a lens share a parent distribution. For example, a single
lens could contain multiple mass components whose parameters are drawn from a common parent distribution. While
no such example is included in the current workspace, the structure shown here could be adapted easily for that case.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autofit as af
from pathlib import Path

"""
__Dataset__

For each lens dataset in our sample we set up the correct path and load it by iterating over a for loop. 

We are loading a different dataset to the previous tutorials, where the lenses only have a single bulge component
which each have different Sersic indexes which are drawn from a parent Gaussian distribution with a mean value 
of 2.0 and sigma of 0.5.

This data is not automatically provided with the autolens workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/advanced/mass_power_law.py`. 
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

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
masked_imaging_list = []

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

    masked_imaging_list.append(dataset)

"""
__Analysis__

For each dataset we now create a corresponding `AnalysisImaging` class, as we are used to doing for `Imaging` data.
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = al.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

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
__Analysis Factors__

Now we have our `Analysis` classes and model components, we can compose our `AnalysisFactor`'s.

These are composed in the same way as for the graphical model and are described in detail in the
`guides/modeling/advanced/graphical` example.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)


"""
__Model__

We now compose the hierarchical model that we fit, using the individual model components created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `slope` of each individual lens mass model is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""
hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.TruncatedGaussianPrior(
        mean=2.0, sigma=1.0, lower_limit=0.0, upper_limit=100.0
    ),
    sigma=af.TruncatedGaussianPrior(
        mean=0.5, sigma=0.5, lower_limit=0.0, upper_limit=100.0
    ),
)

"""
We now add each of the individual mass `slope` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `slope` of every light model in our dataset is now 
assumed to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping 
to estimate.

The code below is not specific to graphical models and could be applied to any model where certain parameters are
assumed to be drawn from a shared parent distribution.
"""
for model in model_list:
    hierarchical_factor.add_drawn_variable(model.galaxies.lens.mass.slope)

"""
__Factor Graph__

We now create the factor graph for this model, using the list of `AnalysisFactor`'s and the hierarchical factor.

Again, this code is described in detail in the `guides/modeling/advanced/graphical` example.
"""
factor_graph = af.FactorGraphModel(
    *analysis_factor_list, hierarchical_factor, use_jax=True
)

"""
The factor graph model `info` attribute shows that the hierarchical factor's parameters are included in the model.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
search = af.Nautilus(
    path_prefix=Path("modeling"),
    name="hierarchical",
    n_live=150,
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows the result, including the hierarchical factor's parameters.
"""
print(result.info)

"""
We can now inspect the inferred value of hierarchical factor's mean and sigma.

We see that they are consistent with the input values of `mean=2.0` and `sigma=0.2`, which are
the values used to simulate the lens dataset sample.
"""
samples = result.samples

mean = samples.median_pdf(as_instance=False)[-2]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-2]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-2]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-2]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-2]

print(
    "Inferred value of the mean of the parent hierarchical distribution for the mass model slopes: \n"
)
print(f"{mean} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{mean} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

scatter = samples.median_pdf(as_instance=False)[-2]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-1]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-1]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-1]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-1]

print(
    "Inferred value of the scatter (the sigma value of the Gassuain) of the parent hierarchical distribution for the mass model slopes: \n"
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

"""
__Concept__

A hierarchical model yields more precise and accurate estimates of the parent distribution’s parameters, but also
the individual parameters fit to each lens. 

This happens because **each dataset informs the shared distribution**, and the distribution in turn constrains 
each individual dataset. This can be described as **“the datasets talking to one another.”**

For example, suppose that when fit alone, `dataset_0` yields a weak constraint on the mass–slope parameter, spanning
1.3 → 2.7 (1σ). Now imagine that, when we include the other datasets, the hierarchical distribution is well constrained
to `mean = 2.0 ± 0.1` and `sigma = 0.10 ± 0.05`. This shared information tells us that values far from ~2.0 are unlikely,
so `dataset_0` will be **forced toward physically plausible solutions**, even though it could not infer this on its own.

In large hierarchical fits with many lenses, this “communication” between datasets can break degeneracies and extract
substantially more information from the sample than independent fits ever could. For inference on parameters like
cosmology, this shrinkage of uncertainties on lens mass model parameters can lead to significantly tighter constraints
on the cosmological parameters themselves.

__Wrap Up__

Hierarchical models enable us to infer **population-level trends** from large lens samples. Using graphical modeling,
we can easily compose complex models with shared parameters and hierarchical structure across many datasets.

However, scaling to large graphs introduces challenges. As models grow in size, poor sampling can lead to local maxima,
and fitting many datasets simultaneously can become computationally expensive (in both CPU time and memory).

The next tutorial introduces **Expectation Propagation (EP)**, a framework that partitions the graphical model into
many small sub-fits—one for each node in the factor graph. Each node fit passes information to its neighbors, allowing
us to fit graphs with hundreds of components and tens of thousands of parameters as a series of manageable, low-dimensional
optimizations.

This makes hierarchical graphical modeling **scalable to the largest datasets and most complex models.**
"""
