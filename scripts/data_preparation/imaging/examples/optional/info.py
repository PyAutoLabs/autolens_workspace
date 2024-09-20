"""
Data Preparation: Info (Optional)
=================================

Auxiliary information about a strong lens dataset may used during an analysis or afterwards when interpreting the
 modeling results. For example, the redshifts of the source and lens galaxy.

By storing these as an `info.json` file in the lens's dataset folder, it is straight forward to load the redshifts
in a modeling script and pass them to a fit, such that **PyAutoLens** can then output results in physical
units (e.g. kpc instead of arc-seconds).

For analysing large quantities of  modeling results, **PyAutoLens** has an sqlite database feature. The info file
may can also be loaded by the database after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to.

For example, to plot the model-results against other measurements of a lens not made by PyAutoLens. Examples of such
data might be:

- The velocity dispersion of the lens galaxy.
- The stellar mass of the lens galaxy.
- The results of previous strong lens models to the lens performed in previous papers.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

# %matplotlib inline

"""
The path where info is output, which is `dataset/imaging/simple__no_lens_light`
"""
dataset_type = "imaging"
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
The info is written as a Python dictionary and can have as many entries as desired added to it. Any information you
want to include int he interpretation of your lens models should be included here.
"""
info = {
    "redshift_lens": 0.5,
    "redshift_source": 1.0,
    "velocity_dispersion": 250000,
    "stellar mass": 1e11,
}

"""
The info is stored in the dataset folder as a .json file. 

We cannot `dump` a .json file using a string which contains a directory, so we dump it to the location of this
script and move it to the appropriate dataset folder. We first delete existing info file in the dataset folder.
"""
import os
import shutil
import json

info_file = "info.json"

with open(info_file, "w+") as f:
    json.dump(info, f, indent=4)

if os.path.exists(path.join(dataset_path, "info.json")):
    os.remove(path.join(dataset_path, "info.json"))

shutil.move("info.json", path.join(dataset_path, "info.json"))

"""
For the info to be available to the results of a model-fit, the modeling script must load the info file from the .json and 
pass it to the search.run() or pipeline.run() function:

info_file = path.join(dataset_path, "info.json")

with open(info_file, "r") as f:
    info = json.load(f)

result = search.fit(model=model, analysis=analysis, info=info)
"""
