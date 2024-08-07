"""
Model Maker: Lens x3 + source x1
================================

For group-scale strong lens modeling, there are multiple lens and / or source galaxies. Manually writing the lens
model of each in the Python scripts we used to perform lens model becomes unfeasible and it better for us to manage
model composition in a separate file and store them in .json files.

This script makes the model that is fitted to the example `group` dataset, where:

 - There are three lens galaxies whose light models are `SersicSph` profiles and total mass distributions
 are `IsothermalSph` models.

 - The source `Galaxy` is modeled as a point source `Point`.

To write your own group-scale lens model, you can easily adapt this script.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
import json
import autofit as af
import autolens as al

"""
__Paths__

The path where the models are output, which is also where the data is stored.
"""
dataset_name = "lens_x3__source_x1"
model_path = path.join("dataset", "group", dataset_name)

os.makedirs(model_path, exist_ok=True)

"""
__Lens_x3__

The group consists of three lens galaxies whose total mass distributions are `IsothermalSph` models.
"""
mass = af.Model(al.mp.IsothermalSph)

mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)

shear = af.Model(al.mp.ExternalShear)

lens_0 = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

mass = af.Model(al.mp.IsothermalSph)
mass.centre_0 = af.GaussianPrior(mean=3.5, sigma=0.5)
mass.centre_1 = af.GaussianPrior(mean=2.5, sigma=0.5)

lens_1 = af.Model(al.Galaxy, redshift=0.5, mass=mass)

mass = af.Model(al.mp.IsothermalSph)
mass.centre_0 = af.GaussianPrior(mean=-4.4, sigma=0.5)
mass.centre_1 = af.GaussianPrior(mean=-5.0, sigma=0.5)

lens_2 = af.Model(al.Galaxy, redshift=0.5, mass=mass)

"""
__Lenses Model__

We now combine the lenses into a `Collection` object and write it to a `.json` file.
"""
lenses = af.Collection(lens_0=lens_0, lens_1=lens_1, lens_2=lens_2)

lenses_file = path.join(model_path, "lenses.json")

with open(lenses_file, "w+") as f:
    json.dump(lenses.dict(), f, indent=4)

"""
__Source_x1__

The group has a single source galaxy whose emission is observed but we model as a `Point`.
"""
point_0 = af.Model(al.ps.Point)
point_0.centre_0 = af.GaussianPrior(mean=0.0, sigma=3.0)
point_0.centre_1 = af.GaussianPrior(mean=0.0, sigma=3.0)

source_0 = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

"""
__Sources Model__

We now combine the source(s) into a `Collection` object and write it to a `.json` file.
"""
sources = af.Collection(source_0=source_0)

sources_file = path.join(model_path, "sources.json")

with open(sources_file, "w+") as f:
    json.dump(sources.dict(), f, indent=4)

"""
Finish.
"""
