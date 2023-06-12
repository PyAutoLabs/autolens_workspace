"""
Database: Subhalo Grid
======================

Dark matter (DM) subhalo analysis can use a grid-search of non-linear searches.

Each cell on this grid fits a DM subhalo whose center is confined to a small 2D segment of the image-plane.

This tutorial shows how to manipulate the results that come out of this grid-search of non-linear searches
via the database.

It follows on from the script `autolens_workspace/*/imaging/results/advanced/result_subhalo_grid.ipynb`, therefore
you should read that first if you have not.
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
__Database__

Load the database. If the file `subhalo_grid.sqlite` does not exist, it will be made by the method below, so its fine 
if you run the code below before the file exists.
"""
database_file = "subhalo_grid.sqlite"

"""
Remove database if you are making a new build (you could delete it manually instead). 

Building the database is slow, so only do this when you redownload results. 
"""
try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

"""
Load the database. If the file `subhalo_grid.sqlite` does not exist, it will be made by the method below, so its fine if
you run the code below before the file exists.
"""
agg = af.Aggregator.from_database(filename=database_file, completed_only=False)

"""
Add all results in the directory "output/results/subhalo_grid" to the database, which we manipulate below via the agg.
Avoid rerunning this once the file `subhalo_grid.sqlite` has been built.
"""
agg.add_directory(directory=path.join("output", "results", "subhalo_grid"))

"""
__Aggregator Grid Search__

By default, the aggregator does not treat the results of a grid-search of non-linear searches in a special way.

Therefore, all 4 (2x2) results on your hard-disk can be accessed via the database using the normal aggregator API.

However, the `grid_searches()` method can be invoked to create an `AggregatorGroup` object which only contains
the results of the grid search and contains bespoke functionality for manipulating them.
"""
agg_grid = agg.grid_searches()

"""
In this example, we fitted only one dataset, therefore the length of the `agg_grid` is 1 and all generators it
create are length 1.
"""
print(len(agg_grid))

"""
We can extract the best-fit results, corresponding to the grid-cell with the highest overall `log_likelihood`.
"""
agg_best_fits = agg_grid.best_fits()

"""
This allows us to make a generator of its best-fit results.
"""
fit_imaging_agg = al.agg.FitImagingAgg(aggregator=agg_best_fits)
fit_imaging_gen = fit_imaging_agg.max_log_likelihood_gen_from()

"""
Because only one dataset was fitted in this example, the length of `fit_imaging_gen` is 1 and the code below visualizes
just one fit.

If fits to multiple dataets were contained in the `output/results/subhalo_grid` directory all of this code would be
sufficient to visualize multiple fits.
"""
for fit in fit_imaging_gen:
    fit_plotter = aplt.FitImagingPlotter(
        fit=fit,
    )
    fit_plotter.subplot_fit()

"""
__Subhalo Result__

The results of a subhalo grid-search use an instance of the `SubhaloResult` class (see 
the `autolens_workspace/*/imaging/results/advanced/result_subhalo_grid.ipynb` tutorial).

This object is made via the aggregator using generators.
"""
for fit_grid, fit_imaging_detect in zip(agg_grid, fit_imaging_gen):
    subhalo_search_result = al.subhalo.SubhaloResult(
        grid_search_result=fit_grid["result"], result_no_subhalo=fit_grid.parent
    )

"""
The tutorial `autolens_workspace/*/imaging/results/advanced/result_subhalo_grid.ipynb` shows examples of manipulating this
object, we show one example below which prints the `subhalo_detection_array` of the subhalo search of every
dataset fitted (in this case just 1 dataset).
"""
for fit_grid, fit_imaging_detect in zip(agg_grid, fit_imaging_gen):
    subhalo_search_result = al.subhalo.SubhaloResult(
        grid_search_result=fit_grid["result"], result_no_subhalo=fit_grid.parent
    )

    subhalo_detection_array = subhalo_search_result.subhalo_detection_array_from(
        use_log_evidences=True, relative_to_no_subhalo=True
    )

    print(subhalo_detection_array)

"""
__Plot__

The `SubhaloPlotter` object can be used for visualizing results via the database.
"""
for fit_grid, fit_imaging_detect in zip(agg_grid, fit_imaging_gen):
    subhalo_search_result = al.subhalo.SubhaloResult(
        grid_search_result=fit_grid["result"], result_no_subhalo=fit_grid.parent
    )

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        subhalo_result=subhalo_search_result,
        fit_imaging_detect=fit_imaging_detect,
        use_log_evidences=True,
    )

    subhalo_plotter.subplot_detection_imaging(remove_zeros=True)
    subhalo_plotter.subplot_detection_fits()
    subhalo_plotter.figure_with_detection_overlay(image=True, remove_zeros=True)

"""
Finish.
"""
