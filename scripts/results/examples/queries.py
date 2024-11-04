"""
Results: Queries
================

Suppose we have the results of many fits in the database and we only wanted to load and inspect a specific set
of model-fits (e.g. the results of `tutorial_1_introduction`). We can use the database's querying tools to only load
the results we are interested in.

The database also supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset)
can be loaded.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al


"""
__Aggregator__

First, set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder"),
)

"""
__Unique Tag__

We can use the `Aggregator` to query the database and return only specific fits that we are interested in. We first 
do this using the `unique_tag` which we can query to load the results of a specific `dataset_name` string we 
input into the model-fit's search. 

By querying using the string `lens_sersic` the model-fit to only the second dataset is returned:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "simple__no_lens_light")
samples_gen = agg_query.values("samples")

"""
As expected, this list now has only 1 `SamplesNest` corresponding to the second dataset.
"""
print("Directory Filtered DynestySampler Samples: \n")
print("Total Samples Objects via unique tag = ", len(list(samples_gen)), "\n\n")

"""
If we query using an incorrect dataset name we get no results:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

"""
__Search Name__

We can also use the `name` of the search used to fit to the model as a query. 

In this example, all three fits used the same search, which had the `name` `database_example`. Thus, using it as a 
query in this example is somewhat pointless. However, querying based on the search name is very useful for model-fits
which use search chaining (see chapter 3 **HowToLens**), where the results of a particular fit in the chain can be
instantly loaded.

As expected, this query contains all 3 results.
"""
name = agg.search.name
agg_query = agg.query(name == "database_example")
print("Total Queried Results via search name = ", len(agg_query), "\n\n")

"""
__Model Queries__

We can also query based on the model fitted. 

For example, we can load all results which fitted an `Isothermal` model-component, which in this simple 
example is all 3 model-fits.

The ability to query via the model is extremely powerful. It enables a user to fit many models to large samples 
of lenses efficiently load and inspect the results. 

[Note: the code `agg.model.galaxies.lens.mass` corresponds to the fact that in the `Model` we named the model components 
`galaxies`, `lens` and `mass`. If the `Model` had used a different name the code below would change correspondingly. 
Models with multiple galaxies are therefore easily accessed via the database.]
"""
lens = agg.model.galaxies.lens
agg_query = agg.query(lens.mass == al.mp.Isothermal)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

"""
We can also query the model on whether a component is None, which was the case for the `disk` we created the source
galaxy using. 

When performing model comparison with search-chaining pipelines, it is common for certain components to be included or 
omitted via a `None`. Querying via `None` therefore allows us to load the results of different model-fits.
"""
source = agg.model.galaxies.source
agg_query = agg.query(source.disk == None)
print("Total Samples Objects via `Isothermal` model query = ", len(agg_query), "\n")

"""
Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the 
inferred value of `sersic_index` for the `Sersic` of the source's bulge is less than 3.0 (which returns only 
the first of the three model-fits).
"""
bulge = agg.model.galaxies.source.bulge
agg_query = agg.query(bulge.sersic_index < 3.0)
print(
    "Total Samples Objects In Query `source.bulge.sersic_index < 3.0` = ",
    len(agg_query),
    "\n",
)

"""
__Logic__

Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted an `Isothermal` mass model AND (using the & symbol) inferred a value of einstein radius above
1.0 for the lens's mass 

The OR logical clause is also supported via the symbol |.
"""
mass = agg.model.galaxies.lens.mass
agg_query = agg.query((mass == al.mp.Isothermal) & (mass.einstein_radius > 1.0))
print(
    "Total Samples Objects In Query `Isothermal and einstein_radius > 3.0` = ",
    len(agg_query),
    "\n",
)

"""
Finished.
"""
