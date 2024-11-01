"""
Database: Introduction
======================

The default behaviour of model-fitting results output is to be written to hard-disc in folders. These are simple to
navigate and manually check.

For small model-fitting tasks this is sufficient, however it does not scale well when performing many model fits to
large datasets, because manual inspection of results becomes time consuming.

All results can therefore be output to an sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database,
meaning that results can be loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation.
This database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can
be loaded.

This script fits a sample of three simulated strong lenses using the same non-linear search. The results will be used
to illustrate the database in the database tutorials that follow.

__Model__

The search fits each lens with:

 - An `Isothermal` `MassProfile` for the lens galaxy's mass.
 - An `Sersic` `LightProfile` for the source galaxy's light.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import autofit as af
import autolens as al

"""
__Unique Identifiers__

Results output to hard-disk are contained in a folder named via a unique identifier (a 
random collection of characters, e.g. `8hds89fhndlsiuhnfiusdh`). The unique identifier changes if the model or 
search change, to ensure different fits to not overwrite one another on hard-disk.

Each unique identifier is used to define every entry of the database as it is built. Unique identifiers therefore play 
the same vital role for the database of ensuring that every set of results written to it are unique.

In this example, we fit 3 different datasets with the same search and model. Each `dataset_name` is therefore passed
in as the search's `unique_tag` to ensure 3 separate sets of results for each model-fit are written to the .sqlite
database.

__Dataset__

For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

We want each results to be stored in the database with an entry specific to the dataset. We'll use the `Dataset`'s name 
string to do this, so lets create a list of the 3 dataset names.
"""
dataset_names = [
    "simple",
    "lens_sersic",
    "mass_power_law",
]

pixel_scales = 0.1

"""
__Results From Hard Disk__

In this example, results will be first be written to hard-disk using the standard output directory structure and we
will then build the database from these results. This behaviour is governed by us inputting `session=None`.

If you have existing results you wish to build a database for, you can therefore adapt this example you to do this.

Later in this example we show how results can also also be output directly to an .sqlite database, saving on hard-disk 
space. This will be acheived by setting `session` to something that is not `None`.
"""
session = None

for dataset_name in dataset_names:
    """
    __Paths__

    Set up the config and output paths.
    """
    dataset_path = path.join("dataset", "imaging", dataset_name)

    """
    __Dataset__

    Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files.

    This `Imaging` object will be available via the aggregator. Note also that we give the dataset a `name` via the
    command `name=dataset_name`. we'll use this name in the aggregator tutorials.
    """
    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=pixel_scales,
    )

    """
    __Mask__

    The `Mask2D` we fit this data-set with, which will be available via the aggregator.
    """
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    """
    __Info__

    Information about the model-fit that is not part included in the model-fit itself can be made accessible via the 
    database by passing an `info` dictionary. 

    Below we write info on the dataset`s (hypothetical) data of observation and exposure time, which we will later show
    the database can access. 

    For fits to large datasets this ensures that all relevant information for interpreting results is accessible.
    """
    with open(path.join(dataset_path, "info.json")) as json_file:
        info = json.load(json_file)

    """
    __Model__

    Set up the model as per usual, and will see in tutorial 3 why we have included `disk=None`.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
            source=af.Model(
                al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore, disk=None
            ),
        )
    )

    """
    The `unique_tag` below uses the `dataset_name` to alter the unique identifier, which as we have seen is also 
    generated depending on the search settings and model. In this example, all three model fits use an identical 
    search and model, so this `unique_tag` is key for ensuring 3 separate sets of results for each model-fit are 
    stored in the output folder and written to the .sqlite database. 
    """
    search = af.Nautilus(
        path_prefix=path.join("database"),
        name="database_example",
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This can instruct the search to write to the .sqlite database.
        n_live=100,
        number_of_cores=6,
        n_like_max=5000,
    )

    analysis = al.AnalysisImaging(dataset=dataset)

    search.fit(analysis=analysis, model=model, info=info)

"""
__Building a Database File From an Output Folder__

The fits above wrote the results to hard-disk in folders, not as an .sqlite database file. 

We build the database below, where the `database_name` corresponds to the name of your output folder and is also the 
name of the `.sqlite` database file that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written to hard-disk (e.g. 
for quick visual inspection) and using the database for sample wide analysis is beneficial.

We can optionally only include completed model-fits but setting `completed_only=True`.

If you inspect the `output` folder, you will see a `database.sqlite` file which contains the results.
"""
database_name = "database"

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
__Writing Directly To Database__

Results can be written directly to the .sqlite database file, skipping output to hard-disk entirely, by creating
a session and passing this to the non-linear search.

The code below shows how to do this, but it is commented out to avoid rerunning the non-linear searches.

This is ideal for tasks where model-fits to hundreds or thousands of datasets are performed, as it becomes unfeasible
to inspect the results of all fits on the hard-disk. 

Our recommended workflow is to set up database analysis scripts using ~10 model-fits, and then scaling these up
to large samples by writing directly to the database.
"""
# session = af.db.open_database("database.sqlite")
#
# search = af.Nautilus(
#     path_prefix=path.join("database"),
#     name="database_example",
#     unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
#     session=session,  # This can instruct the search to write to the .sqlite database.
#     n_live=100,
# )


"""
__Files__

When performing fits which output results to hard-disc, a `files` folder is created containing .json / .csv files of 
the model, samples, search, etc.

These are the files that are written to the database, which the aggregator loads via the database in order to make 
them accessible in a Python script or Jupyter notebook.

You can checkout the output folder created by this fit to see these files.

Below, we will access these results using the aggregator's `values` method. A full list of what can be loaded is
as follows:

 - `model`: The `model` defined above and used in the model-fit (`model.json`).
 - `search`: The non-linear search settings (`search.json`).
 - `samples`: The non-linear search samples (`samples.csv`).
 - `samples_info`: Additional information about the samples (`samples_info.json`).
 - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
 - `info`: The info dictionary passed to the search (`info.json`).
 - `covariance`: The inferred covariance matrix (`covariance.csv`).
 - `cosmology`: The cosmology used by the fit (`cosmology.json`).
 - `settings_inversion`: The settings associated with a inversion if used (`settings_inversion.json`).
 - `dataset/data`: The data that is fitted (`data.fits`).
 - `dataset/noise_map`: The noise-map (`noise_map.fits`).
 - `dataset/psf`: The Point Spread Function (`psf.fits`).
 - `dataset/mask`: The mask applied to the data (`mask.fits`).
 - `dataset/settings`: The settings associated with the dataset (`settings.json`).

The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does reperform calculations using the full 
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the `samples.

__Generators__

Before using the aggregator to inspect results, lets discuss Python generators. 

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects 
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory 
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once. 
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the 
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the 
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the `values` method. This takes the `name` of the
object we want to create a generator of, for example inputting `name=samples` will return the results `Samples`
object.
"""
samples_gen = agg.values("samples")

"""
By converting this generator to a list and printing it, it is a list of 3 `SamplesNest` objects, corresponding to 
the 3 model-fits performed above.
"""
print("Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(agg), "\n")

"""
__Model__

The model used to perform the model fit for each of the 3 datasets can be loaded via the aggregator and printed.
"""
model_gen = agg.values("model")

for model in model_gen:
    print(model.info)

"""
__Search__

The non-linear search used to perform the model fit can be loaded via the aggregator and printed.
"""
search_gen = agg.values("search")

for search in search_gen:
    print(search)

"""
__Samples__

The `Samples` class contains all information on the non-linear search samples, for example the value of every parameter
sampled using the fit or an instance of the maximum likelihood model.

The `Samples` class is described fully in the results cookbook.
"""
for samples in agg.values("samples"):
    print("The tenth sample`s third parameter")
    print(samples.parameter_lists[9][2], "\n")

"""
Therefore, by loading the `Samples` via the database we can now access the results of the fit to each dataset.

For example, we can plot the maximum likelihood model for each of the 3 model-fits performed.
"""
ml_vector = [
    samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
]

print("Max Log Likelihood Model Parameter Lists: \n")
print(ml_vector, "\n\n")

"""
All remaining methods accessible by `agg.values` are described in the other database examples.

__Wrap Up__

This example illustrates how to use the database.

The API above can be combined with the `results/examples` scripts in order to use the database to load results and
perform analysis on them.
"""
