"""
Tutorial Optional: Search Chaining
==================================

The graphical model examples compose a fit individual models to large datasets.

For complex models, one may need to combine graphical models with search chaining in order to ensure that models
are initialized in a robust manner, ensuring automated modeling.

This example script shows how models can be fitted via chaining and output /loaded from to .json files in order
to combine search chaining with graphical models.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/simple__no_lens_light.py`.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from pathlib import Path
import os
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

For each dataset in our sample we set up the correct path and load it by iterating over a for loop. 

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autolens_workspace/scripts/advanced/graphical/simulator/samples/simple__no_lens_light.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple__no_lens_light"

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

We now mask each lens in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every lens in the dataset is appropriate.
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
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = Path("imaging") / "hierarchical" / "tutorial_optional_search_chaining"

"""
__Model__

We compose our model using `Model` objects, which represent the lenses we fit to our data. In this 
example we fit a model where:

 - The lens's bulge is a MGE bulge with 2 x 30 Gaussians with its centre fixed to the input 
 value of (0.0, 0.0) [4 parameters]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLawSph)
lens.mass.centre = (0.0, 0.0)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCoreSph)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit__

For each dataset we now create a non-linear search, analysis and perform the model-fit using this model.

Results are output to a unique folder named using the `dataset_index`.
"""
result_1_list = []

for dataset_index, masked_dataset in enumerate(masked_imaging_list):
    dataset_name_with_index = f"dataset_{dataset_index}"
    path_prefix_with_index = Path(path_prefix, dataset_name_with_index)

    search_1 = af.Nautilus(
        path_prefix=path_prefix,
        name="search[1]__simple__no_lens_light",
        unique_tag=dataset_name_with_index,
        n_live=100,
    )

    analysis_1 = al.AnalysisImaging(dataset=masked_dataset)

    result_1 = search_1.fit(model=model, analysis=analysis_1)
    result_1_list.append(result_1)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens models fitted in search 2, where:

 - The lens's bulge is again a MGE bulge with 2 x 30 Gaussians [6 parameters: priors passed from search 1]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

Prior passing via search chaining is now specific to each result, thus this operates on a list via for loop.
"""
model_2_list = []

for result_1 in result_1_list:
    source = result_1.model.galaxies.source

    mass = af.Model(al.mp.PowerLawSph)
    mass.take_attributes(result_1.model.galaxies.lens.mass)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    model_2_list.append(model_2)

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the `model.info` file of the search 2 model-fit to ensure the priors were passed correctly, as 
well as the checkout the results to ensure an accurate power-law mass model is inferred.
"""
result_2_list = []

for dataset_index, masked_dataset in enumerate(masked_imaging_list):
    dataset_name_with_index = f"dataset_{dataset_index}"
    path_prefix_with_index = Path(path_prefix, dataset_name_with_index)

    search_2 = af.Nautilus(
        path_prefix=path_prefix,
        name="search[2]__mass_sph_pl__source_sersic",
        unique_tag=dataset_name_with_index,
        n_live=100,
    )

    analysis_2 = al.AnalysisImaging(dataset=masked_dataset)

    result_2 = search_2.fit(model=model_2, analysis=analysis_2)
    result_2_list.append(result_2)

"""
__Model Output__

The model can also be output to a .`json` file and loaded in another Python script.

This is not necessary for combining search chaining and graphical models, but can help make scripts shorter if the
search chaining takes up a lot of lines of Python.
"""
model_path = Path("imaging") / "hierarchical" / "models" / "initial"

for dataset_index, model in enumerate(model_2_list):
    model_dataset_path = Path(model_path, f"dataset_{dataset_index}")

    os.makedirs(model_dataset_path, exist_ok=True)

    model_file = Path(model_dataset_path, "model.json")

    with open(model_file, "w") as f:
        json.dump(model.dict(), f, indent=4)


"""
__Model Loading__

We can load the model above as follows.
"""
model_path = Path("imaging") / "hierarchical" / "models" / "initial"

model_list = []

for dataset_index in range(total_datasets):
    model_file = model_path / f"dataset_{dataset_index}" / "model.json"

    model = af.Collection.from_json(file=model_file)

    model_list.append(model)
