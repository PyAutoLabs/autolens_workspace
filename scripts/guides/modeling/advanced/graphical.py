"""
Modeling: Graphical
===================

The example scripts throughout the workspace have focused on modeling **individual strong lens datasets**.
From each fit, you may have inspected lens properties (e.g., Einstein radius) and source properties
(e.g., magnification). You may even have analyzed many lenses one-by-one and combined their results to study
global trends in galaxy formation or cosmology.

However, fitting each lens independently does **not** make full use of the information contained in a large
sample. Many properties are expected to be **shared across lenses** (e.g., population-level mass slopes,
cosmological parameters), and treating them independently ignores this shared structure.

In this example, we demonstrate how to fit **multiple lenses simultaneously** using a **graphical model**.
A graphical model links parameters across separate lens fits, explicitly defining which parameters are unique
to each dataset and which are shared. These links can be arbitrarily complex, enabling joint analysis across
diverse datasets with structured relationships between model components.

Here, we illustrate a cosmological application: inferring the **Hubble constant (H0)** from time-delay lenses.
A graphical model links the mass models across multiple lenses and includes a **shared H0 parameter**, allowing
a joint inference that improves cosmological constraints compared to individual fits.

Graphical models form the foundation of **hierarchical modeling**, where the parameters of individual lenses are
assumed to be drawn from a parent distribution (see `guides/modeling/hierarchical`). Hierarchical approaches can
extract significantly more information from large samples than fitting each dataset independently. The example
shows how the power-law `slope` of each lens's mass distribution is modeled as being drawn from a shared parent
Gaussian distribution, whose hyper-parameters (mean and variance) are inferred from the data.

This example illustrates graphical models using point-source datasets and the Hubble Constant, but it is a clear
and intuitive model whereby a single shared parameter (H0) links multiple lenses. The API and concepts demonstrated
here can be directly applied to imaging and interferometer datasets, and more complex models with many shared can
be composed and fitted using the same framework.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import autolens as al
import autofit as af

"""
__Initialization__

Load 3 simulated time-delay lens datasets which are all simulated with different mass models but 
the same Hubble constant.
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
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1, xp=np
)

"""
__Model__

We compose our model using `Model` objects, which represent the lenses we fit to our data.

This graphical model creates a non-linear parameter space that has parameters for every lens mass and source galaxy point
source in our sample. In this example, there are 3 lenses each with their own model, therefore:

 - The lens galaxy's total mass distribution is an `Isothermal` with fixec centre [3 parameters].

 - The source galaxy's light is a point `Point` [2 parameters].

 - There is a single cosmological shared free parameter, `H0` [1 parameter]

 - There are 3 strong lenses in our graphical model [(3 x 5) + 1 = 16 parameters]. 

The overall dimensionality of parameter space is therefore N=16.
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

For each dataset we now create a corresponding `AnalysisPoint` class.
"""
analysis_list = []

for dataset in dataset_list:
    analysis = al.AnalysisPoint(dataset=dataset, solver=solver)

    analysis_list.append(analysis)

"""
__Analysis Factors__

Above, we created a `model_list` containing three lens models, each sharing the same prior on `H0`.  
We also loaded three datasets and assigned each one an `Analysis` class, which defines how a model
is evaluated against that dataset.

We now pair each model with its corresponding `Analysis` object, telling **PyAutoLens** that:

- `model_list[0]` is fit to `dataset_list[0]` using `analysis_list[0]`
- `model_list[1]` is fit to `dataset_list[1]` using `analysis_list[1]`
- `model_list[2]` is fit to `dataset_list[2]` using `analysis_list[2]`

The point where a `Model` and an `Analysis` meet is called an **`AnalysisFactor`**.

This terminology reflects that we are building a **factor graph**:  
each *factor* corresponds to a node that contains (i) a dataset, (ii) a model for that dataset, and (iii) the
process that fits them together. The links between these nodes define the global structure of the graphical model
we are fitting, including shared parameters such as `H0`.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):

    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We now combine our `AnalysisFactor` objects to form a **factor graph**.

What is a factor graph?  
A factor graph is the explicit representation of our graphical model. It defines:

- the individual model components used to fit each dataset (e.g., the three `Collection` lens + source models), and
- how their parameters are linked or shared (e.g., each lens has its own mass distribution, but all share the same
  cosmological parameter `H0`).

Although PyAutoFit does not yet visualize factor graphs, the conceptual structure is straightforward. A factor graph
consists of:

- **Nodes** — each node corresponds to an `AnalysisFactor`, meaning a specific dataset paired with a model used to fit it.

- **Links** — these represent shared model components or parameters across nodes (e.g., a single `H0` value shared
  across all lenses), ensuring they retain the same value when fitting multiple datasets.

Together, the nodes and links define the full, coupled model that is fit across all datasets simultaneously.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=False)

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
search = af.Nautilus(
    path_prefix=Path("modeling"),
    name="graphical",
    n_live=150,
    n_batch=10,  # GPU batching and VRAM use explained in `modeling` examples.
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows that the result is expressed following the same struture of analysis factors
that the `global_prior_model.info` attribute revealed above.
"""
print(result.info)

"""
__Wrap Up__

The graphical model estimated the Hubble constant by fitting all three lenses **simultaneously**, making full use of
the information shared across the sample.

If you instead fit each lens independently and compute `H0` manually from each result, the combined uncertainty on
`H0` will be larger than the uncertainty from the graphical model. This demonstrates the power of **joint inference**
using graphical models.

Even if you tried to combine the independent fits using importance sampling, you would still not recover the same
precision or accuracy. In addition, the prior on `H0` would be applied three times (once per lens), biasing the
inference.

This example used a simple shared parameter (the Hubble constant) across multiple lenses. The same framework can be
extended to far more complex models, with many shared or linked parameters, enabling powerful hierarchical and
population-level inference for large lens samples.
"""
