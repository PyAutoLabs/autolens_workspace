"""
Results: Data Fitting
=====================

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to perform
fits to the data.

We show how to use these tools to inspect the maximum log likelihood model of a fit to the data, customize things
like its visualization and also inspect fits randomly drawm from the PDF.

__Contents__

**Interferometer:** This script can easily be adapted to analyse the results of charge injection imaging model-fits.
**Aggregator:** First, set up the aggregator as shown in `start_here.py`.
**Fits via Aggregator:** Having performed a model-fit, we now want to interpret and visualize the results.
**Modification:** The `FitImagingAgg` allow us to modify the fit settings.
**Visualization Customization:** The benefit of inspecting fits using the aggregator, rather than the files outputs to the.

__Interferometer__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are:

 - `ImagingAgg` -> `InterferometerAgg`.
 - `FitImagingAgg` -> `FitInterferometerAgg`.
 - `aplt.subplot_imaging_dataset` -> `aplt.subplot_interferometer_dirty_images`.
 - `aplt.subplot_fit_imaging` -> `aplt.subplot_fit_interferometer`.

Quantities specific to an interfometer, for example its uv-wavelengths real space mask, are accessed using the same API
(e.g. `values("dataset.uv_wavelengths")` and `.values{"dataset.real_space_mask")).
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import os
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Aggregator__

First, set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

results_path = Path("output") / "results_folder"
if not results_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/guides/results/_quick_fit.py"],
        check=True,
    )

agg = Aggregator.from_directory(
    directory=Path("output") / "results_folder",
)

"""
The masks we used to fit the lenses is accessible via the aggregator.
"""
mask_gen = agg.values("dataset.mask")
print([mask for mask in mask_gen])

"""
The info dictionary we passed is also available.
"""
print("Info:")
info_gen = agg.values("info")
print([info for info in info_gen])

"""
__Fits via Aggregator__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we inspect 
the `Imaging` objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Imaging` object. For large datasets, this would require us to use generators to ensure it is 
memory-light, which are cumbersome to write.

This example therefore uses the `ImagingAgg` object, which conveniently loads the `Imaging` objects of every fit via 
generators for us. 

We get a dataset generator via the `al.agg.ImagingAgg` object, where this `dataset_gen` contains the maximum log
likelihood `Imaging `object of every model-fit.

The `dataset_gen` returns a list of `Imaging` objects, as opposed to just a single `Imaging` object. This is because
only a single `Analysis` class was used in the model-fit, meaning there was only one `Imaging` dataset that was
fit. 

The `multi` package of the workspace illustrates model-fits which fit multiple datasets 
simultaneously, (e.g. multi-wavelength imaging)  by summing `Analysis` objects together, where the `dataset_list` 
would contain multiple `Imaging` objects.
"""
dataset_agg = al.agg.ImagingAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

for dataset_list in dataset_gen:
    # Only one `Analysis` so take first and only dataset.
    dataset = dataset_list[0]

    aplt.subplot_imaging_dataset(dataset=dataset)

"""
We now use the aggregator to load a generator containing the fit of the maximum log likelihood model (and therefore 
fit) to each dataset.

Analogous to the `dataset_gen` above returning a list with one `Imaging` object, the `fit_gen` returns a list of
`FitImaging` objects, because only one `Analysis` was used to perform the model-fit.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    aplt.subplot_fit_imaging(fit=fit)

"""
__Modification__

The `FitImagingAgg` allow us to modify the fit settings. 

However, we can change these settings such that the fit is performed differently. For example, what if I wanted to see 
how the fit looks where the pixelization didn`t use a border? 

You can do this by passing the settings objects, which overwrite the ones used by the analysis.
"""
fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings=al.Settings(use_border_relocator=False),
)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    aplt.subplot_fit_imaging(fit=fit)

"""
__Visualization Customization__

The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the `plot_yx` and `plot_array`/`subplot_\*` objects..

We create a new function to apply as a generator to do this. However, we use a convenience method available 
in the aggregator package to set up the fit.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")

"""
Making this plot for a paper? You can output it to hard disk.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]


"""
__Errors (Random draws from PDF)__

In the `examples/models.py` example we showed how `Tracer objects could be randomly drawn form the Probability 
Distribution Function, in order to quantity things such as errors.

The same approach can be used with `FitImaging` objects, to investigate how the properties of the fit vary within
the errors (e.g. showing source reconstructions fot different fits consistent with the errors).
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)


for fit_list_gen in fit_gen:  # 1 Dataset so just one fit
    for (
        fit_list
    ) in (
        fit_list_gen
    ):  #  Iterate over each total_samples=2, each with one fits for 1 analysis.
        # Only one `Analysis` so take first and only dataset.
        fit = fit_list[0]

    aplt.subplot_fit_imaging(fit=fit)

"""
Finished.
"""
