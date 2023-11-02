"""
Results: Subhalo Grid
=====================

Dark matter (DM) subhalo analysis can use a grid-search of non-linear searches.

Each cell on this grid fits a DM subhalo whose center is confined to a small 2D segment of the image-plane.

This tutorial shows how to manipulate the results that come out of this grid-search of non-linear searches,
including:

 - Visualization showing how much in each grid cell adding a DM subhalo to the model increases the Bayesian evidence
   compared to a lens model without a DM Subhalo.

 - Tools for comparing the results of models with and without a DM subhalo.

__Subhalo Fit__

The standard DM subhalo analysis in **PyAutoLens** is performed in three stages:

 - A model-fit of a model without a DM subhalo.
 - A model-fit using the grid search of non-linear searches.
 - A model fit where the highest log likelihood DM subhalo result in the grid search is refit, in order to provide a
   better estimate of the Bayesian evidence.

This result script begins by performing the above 3 stages to set up the results. To ensure things run fast:

 - A parametric source is fitted.
 - Only a 2x2 grid search of non-linear searches is performed.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "dark_matter_subhalo"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Fit (Search 1)__

Fit a lens model without a DM Subhalo.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=al.mp.Isothermal,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search_1 = af.Nautilus(
    path_prefix=path.join("results", "subhalo_grid"),
    name="search[1]_no_subhalo",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

Search 2E we perform a [number_of_steps x number_of_steps] grid search of non-linear searches where:

 - The lens galaxy mass is modeled using result 1's mass distribution [Priors initialized from result_1].
 - The source galaxy's light is parametric using result 1 [Model and priors initialized from result_1].
 - The subhalo redshift is fixed to that of the lens galaxy.
 - Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
 - The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

This search aims to detect a dark matter subhalo.
"""
subhalo = af.Model(al.Galaxy, mass=af.Model(al.mp.NFWMCRLudlowSph))

subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)
subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-3.0, upper_limit=3.0)

subhalo.redshift = result_1.instance.galaxies.lens.redshift
subhalo.mass.redshift_object = result_1.instance.galaxies.lens.redshift

subhalo.mass.redshift_source = result_1.instance.galaxies.source.redshift

model = af.Collection(
    galaxies=af.Collection(
        lens=result_1.model.galaxies.lens,
        subhalo=subhalo,
        source=result_1.model.galaxies.source,
    ),
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

search = af.Nautilus(
    path_prefix=path.join("results", "subhalo_grid"),
    name=f"subhalo[2]_subhalo_search]",
    n_live=100,
    number_of_cores=1,
    force_x1_cpu=True,  # ensures parallelizing over grid search works.
)

subhalo_grid_search = af.SearchGridSearch(
    search=search, number_of_steps=2, number_of_cores=1
)


"""
Note the input of the `parent` below, which links the subhalo grid search to the previous lens model fitted.

This links the Bayesian evidence of the model fitted in search 1 to the subhalo grid search, making certain
visualizaiton easier.
"""
grid_search_result = subhalo_grid_search.fit(
    model=model,
    analysis=analysis_2,
    grid_priors=[
        model.galaxies.subhalo.mass.centre_1,
        model.galaxies.subhalo.mass.centre_0,
    ],
    parent=search_1,  # This will be used in the result generation below
)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

Search 3 we refit the lens and source models above but now including a subhalo, where the subhalo model is 
initialized from the highest evidence model of the subhalo grid search.

 - The lens galaxy mass is modeled using result_2's mass distribution [Priors initialized from result_2].
 - The source galaxy's light is parametric [Model and priors initialized from result_2].
 - The subhalo redshift is fixed to that of the lens galaxy.
 - The grid search varies the subhalo (y,x) coordinates and mass as free parameters.
 - The priors on these (y,x) coordinates are GaussianPriors, corresponding to the best-fit grid cell in the 
   grid search performed above.

This search aims to refine the parameter estimates and errors of a dark matter subhalo detected in the grid search
above.
"""
subhalo = af.Model(
    al.Galaxy,
    redshift=result_1.instance.galaxies.lens.redshift,
    mass=af.Model(al.mp.NFWMCRLudlowSph),
)

subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
subhalo.mass.centre = grid_search_result.model_absolute(
    a=1.0
).galaxies.subhalo.mass.centre

subhalo.redshift = grid_search_result.model.galaxies.subhalo.redshift
subhalo.mass.redshift_object = subhalo.redshift

model = af.Collection(
    galaxies=af.Collection(
        lens=grid_search_result.model.galaxies.lens,
        subhalo=subhalo,
        source=grid_search_result.model.galaxies.source,
    ),
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

search = af.Nautilus(
    path_prefix=path.join("results", "subhalo_grid"),
    name=f"subhalo[3]_subhalo_refine",
    n_live=150,
)

result_3 = search.fit(model=model, analysis=analysis_3)

"""
__Subhalo Result__

The results of a subhalo grid-search are returned as an instance of the `SubhaloResult` class.
"""
subhalo_result = al.subhalo.SubhaloResult(
    grid_search_result=grid_search_result,
    fit_agg_no_subhalo=result_1,
)

print(subhalo_result)

"""
This object has built-in arrays containing the key results of the subhalo grid search.

For example, the function `subhalo_detection_array_from()` returns an `Array2D` object containing the Bayesian 
evidence of every model fitted by the subhalo grid.
"""
subhalo_detection_array = subhalo_result.detection_array_from(
    use_log_evidences=True, relative_to_no_subhalo=True
)

"""
In the code above, the input `relative_to_no_subhalo=True` means that every Bayesian evidence is returned as its
difference from the Bayesian evidence inferred for the lens model without a DM subhalo in search 1.

This is possible because of the input `parent=search_1` of the function `subhalo_grid_search.fit()`.

If we set `relative_to_no_subhalo=False` the actual Bayesian evidence inferred in search 2 is returned instead.
"""
subhalo_detection_array = subhalo_result.detection_array_from(
    use_log_evidences=True, relative_to_no_subhalo=False
)

"""
The maximum log likelihood subhalo mass inferred by every fit is also accessible.
"""
subhalo_mass_array = subhalo_result.subhalo_mass_array_from()

"""
The `FitImaging` of search 1, which did not use a DM subhalo, is also available. 

This is used for some of the visualization below. 
"""
print(subhalo_result.fit_imaging_before)

"""
__Plot__

The `SubhaloPlotter` object contains convenience methods for visualizing these results.
"""
subhalo_plotter = aplt.SubhaloPlotter(
    subhalo_result=subhalo_result,
    fit_imaging_detect=result_3.max_log_likelihood_fit,
    use_log_evidences=True,
)

"""
A plot of the lensed source model, with the subhalo grid search overlaid, is produced via:
"""
subhalo_plotter.figure_with_detection_overlay()

"""
The mass overlay is given as follows:
"""
subhalo_plotter.figure_with_mass_overlay()

"""
A subhalo summarizing the detection:
"""
subhalo_plotter.subplot_detection_imaging()

"""
A subplot comparing the fit with and without a DM subhalo is given as follows:
"""
subhalo_plotter.subplot_detection_fits()

"""
Finish.
"""
