"""
Tutorial 2: Graphical Models
============================

In the previous tutorial we fitted individual lens models to individual lens datasets. We knew that every lens in
the dataset had same value of `slope`, which we estimated using the weighted average of the slopes inferred for each 
lens fit.

Graphical modeling follows a different approach. It composes a single model that is fitted to the entire lens dataset.
This model includes specific model component for every individual lens in the sample. However, the graphical
model also has shared parameters between these individual lens models.

This example fits a graphical model using the same sample fitted in the previous tutorial, consisting of
imaging data of three lenses. We fit the `PowerLawSph` plus `SphExpoenntial` model to each lens and source galaxy.
However, whereas previously the `slope` of each lens model component was a free parameter in each fit, in the 
graphical model there is only a single value of `slope` shared by all three lenses (which is how the galaxy data was 
simulated).

This graphical model creates a non-linear parameter space that has parameters for every lens in our sample. In this
example, there are 3 lenses each with their own lens model, therefore:

 - Each lens has 1 free parameter from the components of its `SphIsoterhaml` that are not 
 shared (the `einstein_radius` paramrters).

 - Each source has 4 free parameters for their `ExponentialSph` components.

- There are three lenses and source in total, giving [3 x 1 + 3 x 4 = 16 free parameters]

 - There is one additional free parameter, which is the `slope` shared by all 3 lenses.

The overall dimensionality of parameter space is therefore N=17.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/simulators/imaging/samples/simple__no_lens_light.py`.

__Realism__

For an realistic lens sample, one would not expect that each lens has the same value of `slope`, as is
assumed in tutorials 1, 2 and 3. We make this assumption here to simplify the problem and make it easier to
illustrate graphical models. Later tutorials fit more realistic graphical models where each lens has its own value of
slope!
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

For each galaxy dataset in our sample we set up the correct path and load it by iterating over a for loop. 

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
__Model__

We first set up a shared prior for `slope` which will be attached to the mass profile of every model lens.

By overwriting their `slope` parameters in this way, only one `slope` parameter shared across the whole 
model is used.
"""
slope_shared_prior = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)

"""
__Model__

We compose our model using `Model` objects, which represent the lenses we fit to our data.

This graphical model creates a non-linear parameter space that has parameters for every lens and source galaxy in our 
sample. In this example, there are 3 lenses each with their own model, therefore:

 - Each lens galaxy's total mass distribution is an `PowerLawSph` with its centre fixed to its true value of 
 (0.0, 0.0) [1 parameter].
 
 - Each source galaxy's light is a linear parametric `ExponentialSph` [3 parameters].

 - There are three lenses in our graphical model [3 x 1 = 3 parameters]. 

 - There are three source in our graphical model [3 x 4 = 12 parameters]. 

 - There is one additional free parameter, which is the `slope` shared by all 3 lenses.

The overall dimensionality of parameter space is therefore N=16.
"""
model_list = []

for model_index in range(total_datasets):
    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
    lens.mass.centre = (0.0, 0.0)

    # This makes every Galaxy share the same `slope`.
    lens.mass.slope = slope_shared_prior

    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    model_list.append(model)

"""
__Analysis__

For each dataset we now create a corresponding `AnalysisImaging` class, as we are used to doing for `Imaging` data.
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = al.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

"""
__Analysis Factors__

Above, we composed a `model_list` consisting of three lens models which each had a shared `slope` prior. We 
also loaded three datasets which we intend to fit with each of these lens models, setting up each in an `Analysis` 
class that defines how the model is used to fit the data.

We now simply pair each lens model to each `Analysis` class, so that **PyAutoLens** knows that: 

- `model_list[0]` fits `masked_imaging_list[0]` via `analysis_list[0]`.
- `model_list[1]` fits `masked_imaging_list[1]` via `analysis_list[1]`.
- `model_list[2]` fits `masked_imaging_list[2]` via `analysis_list[2]`.

The point where a `Model` and `Analysis` class meet is called an `AnalysisFactor`. 

This term is used to denote that we are composing a graphical model, which is commonly termed a 'factor graph'. A 
factor defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between 
these different nodes then define the global model we are fitting.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)


"""
__Factor Graph__

We combine our `AnalysisFactor`'s to compose a factor graph.

What is a factor graph? A factor graph defines the graphical model we have composed. For example, it defines the 
different model components that make up our model (e.g. the three `Collection` objects containing the lens and source
galaxies) and how their parameters are linked or shared (e.g. that each `PowerLawSph` has its own unique parameters 
but a shared `slope` parameter).

This is what our factor graph looks like (visualization of graphs not implemented in **PyAutoFit** yet): 

The factor graph above is made up of two components:

- Nodes: these are points on the graph where we have a unique set of data and a model that is made up of a subset of 
our overall graphical model. This is effectively the `AnalysisFactor` objects we created above. 

- Links: these define the model components and parameters that are shared across different nodes and thus retain the 
same values when fitting different datasets.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
The fit will use the factor graph's `global_prior_model`, which uses the models contained in every analysis factor 
to contrast the overall global model that is fitted.

Printing the `info` attribute of this model reveals the overall structure of the model, which is grouped in terms
of the analysis factors and therefore datasets.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
Nautilus = af.Nautilus(
    path_prefix=path_prefix,
    name="tutorial_2_graphical_models",
    n_live=150,
)

result = Nautilus.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same struture of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
We can now inspect the inferred value of `slope`, and compare this to the value we estimated in the previous tutorial
via a weighted average.

(The errors of the weighted average below is what was estimated for a run on my PC, yours may be slightly 
different!)
"""
print(
    f"Weighted Average slope Estimate = 1.996152957641609 (0.02161402431870052) [1.0 sigma confidence intervals] \n"
)

slope = result.samples.median_pdf()[0].galaxies.lens.mass.slope

u1_error = result.samples.values_at_upper_sigma(sigma=1.0)[0].galaxies.lens.mass.slope
l1_error = result.samples.values_at_lower_sigma(sigma=1.0)[0].galaxies.lens.mass.slope

u3_error = result.samples.values_at_upper_sigma(sigma=3.0)[0].galaxies.lens.mass.slope
l3_error = result.samples.values_at_lower_sigma(sigma=3.0)[0].galaxies.lens.mass.slope

print("Inferred value of the shared slope via a graphical model fit: \n")
print(f"{slope} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{slope} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

"""
__Wrap Up__

The graphical model's slope estimate and errors are pretty much exactly the same as the weighted average!

Whats the point of fitting a graphical model if the much simpler approach of the previous tutorial gives the
same answer? 

The answer, is model complexity. Graphical models become more powerful as we make our model more complex,
our non-linear parameter space higher dimensionality and the degeneracies between different parameters on the graph
more significant. 

We will demonstrate this in the next tutorial.

__Wrap Up__

In this tutorial, we showed that for our extremely simple model the graphical model gives pretty much the
same estimate of the lens mass model slope's as simpler approaches followed in the previous tutorial. 

We will next show the strengths of graphical models by fitting more complex models.
"""
