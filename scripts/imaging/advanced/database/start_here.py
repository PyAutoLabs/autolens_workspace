"""
Database: Introduction
======================

The default behaviour of **PyAutoLens** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users 
have a need to perform many model fits to large samples of lenses, making manual inspection of results time consuming.

PyAutoLens's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This script fits a sample of three simulated strong lenses using the same non-linear search. The results will be used
to illustrate the database in the database tutorials that follow.

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
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = af.db.open_database("database.sqlite")

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

    The `SettingsImaging` (which customize the fit of the search`s fit), will also be available to the aggregator! 
    """
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    settings_dataset = al.SettingsImaging(grid_class=al.Grid2D, sub_size=1)

    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_settings(settings=settings_dataset)

    """
    __Info__

    Information about our model-fit that isn't part of the model-fit can be made accessible to the database, by 
    passing an `info` dictionary. 

    Below we load this info dictionary from an `info.json` file stored in each dataset's folder. This dictionary
    contains the (hypothetical) lens redshift, source redshift and lens velocity dispersion of every lens in our sample.
    """
    with open(path.join(dataset_path, "info.json")) as json_file:
        info = json.load(json_file)

    """
    __Pickle Files__

    We can pass strings specifying the path and filename of .pickle files stored on our hard-drive to the `search.fit()`
    method, which will make them accessible to the aggregator to aid interpretation of results. Our simulated strong
    lens datasets have a `true_tracer.pickle` file which we pass in below, which we use in the `Aggregator` tutorials 
    to check if the model-fit recovers its true input parameters.
    """
    pickle_files = [path.join(dataset_path, "true_tracer.pickle")]

    """
    __Model__
    
    Set up the model as per usual, and will see in tutorial 3 why we have included `disk=None`.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic, disk=None),
        )
    )

    """
    In all examples so far, results were written to the `autofit_workspace/output` folder with a path and folder 
    named after a unique identifier, which was derived from the non-linear search and model. This unique identifier
    plays a vital role in the database: it is used to ensure every entry in the database is unique. 

    In this example, results are written directly to the `database.sqlite` file after the model-fit is complete and 
    only stored in the output folder during the model-fit. This can be important for performing large model-fitting 
    tasks on high performance computing facilities where there may be limits on the number of files allowed, or there
    are too many results to make navigating the output folder manually feasible.

    The `unique_tag` below uses the `dataset_name` to alter the unique identifier, which as we have seen is also 
    generated depending on the search settings and model. In this example, all three model fits use an identical 
    search and model, so this `unique_tag` is key for ensuring 3 separate sets of results for each model-fit are 
    stored in the output folder and written to the .sqlite database. 
    """
    search = af.DynestyStatic(
        path_prefix=path.join("database"),
        name="database_example",
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This instructs the search to write to the .sqlite database.
        nlive=50,
    )

    analysis = al.AnalysisImaging(dataset=dataset)

    search.fit(analysis=analysis, model=model, info=info, pickle_files=pickle_files)

"""
If you inspect the `autolens_workspace/output/database` folder during the model-fit, you'll see that the results
are only stored there during the model fit, and they are written to the database and removed once complete. 

__Loading Results__

After fitting a large suite of data, we can use the aggregator to load the database's results. We can then
manipulate, interpret and visualize them using a Python script or Jupyter notebook.

The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
database_file = "database.sqlite"
agg = af.Aggregator.from_database(filename=database_file)

"""
__Generators__

Before using the aggregator to inspect results, let me quickly cover Python generators. A generator is an object that 
iterates over a function when it is called. The aggregator creates all of the objects that it loads from the database 
as generators (as opposed to a list, or dictionary, or other Python type).

Why? Because lists and dictionaries store every entry in memory simultaneously. If you fit many datasets, this will use 
a lot of memory and crash your laptop! On the other hand, a generator only stores the object in memory when it is used; 
Python is then free to overwrite it afterwards. Thus, your laptop won't crash!

There are two things to bare in mind with generators:

 1) A generator has no length and to determine how many entries it contains you first must turn it into a list.

 2) Once we use a generator, we cannot use it again and need to remake it. For this reason, we typically avoid 
 storing the generator as a variable and instead use the aggregator to create them on use.

We can now create a `samples` generator of every fit. The `results` example scripts show how  
the `Samples` class acts as an interface to the results of the non-linear search.
"""
samples_gen = agg.values("samples")

"""
When we print this the length of this generator converted to a list of outputs we see 3 different `SamplesDynesty`
instances. 

These correspond to each fit of each search to each of our 3 images.
"""
print("NestedSampler Samples: \n")
print(samples_gen)
print()
print("Total Samples Objects = ", len(agg), "\n")

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
__Building a Database File From an Output Folder__

The fits above directly wrote the results to the .sqlite file, which we loaded above. However, you may have results
already written to hard-disk in an output folder, which you wish to build your .sqlite file from.

This can be done via the following code, which is commented out below to avoid us deleting the existing .sqlite file.

Below, the `database_name` corresponds to the name of your output folder and is also the name of the `.sqlite` file
that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written
to hard-disk (e.g. for quick visual inspection) but using the database for sample-wide analysis may be benefitial.
"""
# database_name = "database"

# agg = af.Aggregator.from_database(
#    filename=f"{database_name}.sqlite", completed_only=False
# )

# agg.add_directory(directory=path.join("output", database_name)))

"""
__Wrap Up__

This example illustrates how to use the database.

The `database/examples` folder contains examples illustrating the following:

- ``samples.py``: Loads the non-linear search results from the SQLite3 database and inspect the 
   samples (e.g. parameter estimates, posterior).
   
- ``queries.py``: Query the database to get certain  modeling results (e.g. all lens models where `
   einstein_radius > 1.0`).

- ``models.py``: Inspect the models in the database (e.g. visualize their deflection angles).

- ``data_fitting.py``: Inspect the data-fitting results in the database (e.g. visualize the residuals).
"""
