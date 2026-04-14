"""
Subhalo Detection Database: Group
==================================

The example ``subhalo/detect/start_here.py`` shows how to perform dark matter subhalo detection in a group-scale
strong lens with **PyAutoLens**, including using results to inspect and visualize the fit.

This example shows how to load the results of a group-scale subhalo detection analysis into a ``.sqlite`` database,
which can be manipulated stand-alone in this Python script or in a Jupyter notebook. This is useful when fits are
performed on a super computer and results are downloaded separately for inspection.

The database in this example is built by scraping the results of the ``subhalo/detect/start_here.py`` example. You
can also write results directly to the database during the fit by using a session.

__Contents__

**Start Here Notebooks:** You should be familiar with dark matter subhalo detection for group-scale lenses.
**Grid Searches:** Querying the grid search results from the database.
**Grid Search Visualization:** Plotting the grid search results.
**Best Fit:** Retrieving the best-fit results from the grid search.

__Start Here Notebooks__

You should be familiar with dark matter subhalo detection for group-scale lenses, by reading the example
``subhalo/detect/start_here.py``.

You should also be familiar with the database, by reading the example ``imaging/advanced/database/start_here.ipynb``.

__Group Adaptation__

For group-scale lenses, the database contains results that include all group galaxy parameters (main lens
galaxies and extra galaxies) alongside the subhalo parameters. This allows comprehensive inspection of how
the group mass model interacts with the subhalo detection.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import json
import os
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Database__

The name of the database, which corresponds to the output results folder.
"""
database_name = "subhalo_detect_group"

"""
If the ``.sqlite`` file of the database is already in the output folder we delete it and create a new database
immediately afterwards.

This ensures we don't double up on results if we run the script multiple times, and if new results are added to the
output folder (e.g. downloaded from a super computer) they are added to the database.
"""
try:
    os.remove(Path("output", f"{database_name}.sqlite"))
except FileNotFoundError:
    pass

"""
Create the database file ``subhalo_detect_group.sqlite`` in the output folder.
"""
agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

"""
Add all results in the directory "output" to the database, which we manipulate below via the aggregator.
"""
agg.add_directory(directory=Path("output", database_name))

"""
__Agg No / With Subhalo__

Standard aggregator querying can be used to get aggregates of results for lens models with and without a subhalo.

The easiest query uses the name of the subhalo searches in the SLaM subhalo pipeline.
"""
agg_no_subhalo = agg.query(agg.search.name == "subhalo[1]")
agg_with_subhalo = agg.query(agg.search.name == "subhalo[3]_[single_plane_refine]")

"""
We can extract the ``log_evidence`` values of the results with and without a DM subhalo via the aggregators.

We create a dictionary of these values where the keys are the ``unique_tag`` of each search, which is the name of
the dataset fitted.
"""
log_evidence_no_subhalo_dict = {}

for search, samples in zip(
    agg_no_subhalo.values("search"), agg_no_subhalo.values("samples")
):
    log_evidence_no_subhalo_dict[search.unique_tag] = samples.log_evidence

print("\nLog Evidence No Subhalo")
print(log_evidence_no_subhalo_dict)

log_evidence_with_subhalo_dict = {}

for search, samples in zip(
    agg_with_subhalo.values("search"), agg_with_subhalo.values("samples")
):
    log_evidence_with_subhalo_dict[search.unique_tag] = samples.log_evidence

print("\nLog Evidence With Subhalo")
print(log_evidence_with_subhalo_dict)

log_evidence_difference_dict = {}

for key in log_evidence_no_subhalo_dict.keys():
    if key in log_evidence_with_subhalo_dict:
        log_evidence_difference_dict[key] = (
            log_evidence_with_subhalo_dict[key] - log_evidence_no_subhalo_dict[key]
        )

print("\nLog Evidence Difference")
print(log_evidence_difference_dict)

"""
From these, we can create the maximum likelihood instances of the lens model and corresponding ``FitImaging`` objects.

These can then be passed to visualization functions to inspect the results.

For group-scale lenses, the fit objects contain the full group model including all main lens galaxies and extra
galaxies, allowing inspection of how each galaxy's parameters interact with the subhalo detection.
"""
fit_agg_no_subhalo = al.agg.FitImagingAgg(aggregator=agg_no_subhalo)
fit_no_subhalo_gen = fit_agg_no_subhalo.max_log_likelihood_gen_from()
fit_no_subhalo = list(fit_no_subhalo_gen)[0]

fit_agg_with_subhalo = al.agg.FitImagingAgg(aggregator=agg_with_subhalo)
fit_with_subhalo_gen = fit_agg_with_subhalo.max_log_likelihood_gen_from()
fit_with_subhalo = list(fit_with_subhalo_gen)[0]

aplt.subplot_detection_fits(
    fit_imaging_no_subhalo=fit_no_subhalo,
    fit_imaging_with_subhalo=fit_with_subhalo,
)

"""
__Grid Searches__

If the results of the database include a grid search of non-linear searches, the aggregator has a dedicated method
to return the grid of results.

We iterate over these results using a for loop below, where each iteration will correspond to a different lens
in our analysis (e.g. if there are multiple lenses in the dataset that are fitted). In this example, only one
lens is fitted, so this for loop is only iterated over once.

For group-scale lenses, the grid search results include all group galaxy parameters alongside the subhalo
parameters.
"""
for agg_grid, search in zip(
    agg.grid_searches(), agg.grid_searches().best_fits().values("search")
):
    result_subhalo_grid_search = agg_grid["result"]

    result_subhalo_grid_search = al.subhalo.SubhaloGridSearchResult(
        result=result_subhalo_grid_search
    )

    log_evidence_array = result_subhalo_grid_search.figure_of_merit_array(
        use_log_evidences=True,
        relative_to_value=log_evidence_no_subhalo_dict[search.unique_tag],
    )

    print(log_evidence_array)

"""
__Grid Search Visualization__

The grid search visualization tools can also be used to plot the results of the grid search.
"""
samples_no_subhalo_gen = agg_no_subhalo.values("samples")

fit_agg_no_subhalo = al.agg.FitImagingAgg(aggregator=agg_no_subhalo)
fit_no_subhalo_gen = fit_agg_no_subhalo.max_log_likelihood_gen_from()

fit_agg_with_subhalo = al.agg.FitImagingAgg(aggregator=agg_with_subhalo)
fit_with_subhalo_gen = fit_agg_with_subhalo.max_log_likelihood_gen_from()

for agg_grid, fit_no_subhalo, fit_with_subhalo, samples_no_subhalo in zip(
    agg.grid_searches(),
    fit_no_subhalo_gen,
    fit_with_subhalo_gen,
    samples_no_subhalo_gen,
):
    result_subhalo_grid_search = agg_grid["result"]

    result_subhalo_grid_search = al.subhalo.SubhaloGridSearchResult(
        result=result_subhalo_grid_search
    )

    aplt.subplot_detection_imaging(
        result=result_subhalo_grid_search,
        fit_imaging_with_subhalo=fit_with_subhalo,
    )
    aplt.subplot_detection_fits(
        fit_imaging_no_subhalo=fit_no_subhalo,
        fit_imaging_with_subhalo=fit_with_subhalo,
    )


"""
__Best Fit__

We can retrieve a new aggregator containing only the maximum log evidence results of the grid search.

This can then be used as a normal aggregator to inspect the ``Samples`` of the fit or plot the best-fit
``FitImaging``.
"""
agg_best_fit = agg.grid_searches().best_fits()

samples_gen = agg_best_fit.values("samples")

for samples in samples_gen:
    print(samples.log_evidence)

fit_agg = al.agg.FitImagingAgg(
    aggregator=agg_best_fit,
)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    fit = fit_list[0]

    aplt.subplot_fit_imaging(fit=fit)
