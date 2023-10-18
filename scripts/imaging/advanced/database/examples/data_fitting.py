"""
Database: Data Fitting
======================

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to perform
fits to the data.

We show how to use these tools to inspect the maximum log likelihood model of a fit to the data, customize things
like its visualization and also inspect fits randomly drawm from the PDF.

__Interferometer__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are: 

 - `ImagingAgg` -> `InterferometerAgg`.
 - `FitImagingAgg` -> `FitInterferometerAgg`.
 - `Clocker1D` -> `Clocker2D`.
 - `SettingsImaging` -> `SettingsInterferometer`.
 - `ImagingPlotter` -> `InterferometerPlotter`.
 - `FitImagingPlotter` -> `FitInterferometerPlotter`.

Quantities specific to an interfometer, for example its uv-wavelengths real space mask, are accessed using the same API
(e.g. `values("dataset.uv_wavelengths")` and `.values{"dataset.real_space_mask")).
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database("database.sqlite")

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
__Fits via Database__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we inspect 
the `Imaging` object objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `Imaging` object. For large datasets, this would require us to use generators to ensure it is 
memory-light, which are cumbersome to write.

This example therefore uses the `ImagingAgg` object, which conveniently loads the `Imaging` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a dataset generator via the `ac.agg.ImagingAgg` object, where this `dataset_gen` contains the maximum log
likelihood `Imaging `object of every model-fit.

The `dataset_gen` returns a list of `Imaging` objects, as opposed to just a single `Imaging`object. This is because
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

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
We now use the database to load a generator containing the fit of the maximum log likelihood model (and therefore 
fit) to each dataset.

Analogous to the `dataset_gen` above returning a list with one `Imaging` object, the `fit_gen` returns a list of
`FitImaging` objects, because only one `Analysis` was used to perform the model-fit.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

"""
__Modification__

The `FitImagingAgg` allow us to modify the fit settings. By default, it uses the `SettingsImaging`, 
`SettingsPixelization` and `SettingsInversion` that were used during the model-fit. 

However, we can change these settings such that the fit is performed differently. For example, what if I wanted to see 
how the fit looks where the `Grid2D`'s `sub_size` is 4 (instead of the value of 2 that was used)? Or where the 
pixelization didn`t use a border? 

You can do this by passing the settings objects, which overwrite the ones used by the analysis.
"""
fit_agg = al.agg.FitImagingAgg(
    aggregator=agg,
    settings_dataset=al.SettingsImaging(sub_size=4),
    settings_pixelization=al.SettingsPixelization(use_border=False),
)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

"""
__Visualization Customization__

The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the `MatPlot1D` and `MatPlot2D` objects..

Below, we create a new function to apply as a generator to do this. However, we use a convenience method available 
in the aggregator package to set up the fit.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    mat_plot = aplt.MatPlot2D(
        figure=aplt.Figure(figsize=(12, 12)),
        title=aplt.Title(label="Custom Image", fontsize=24),
        yticks=aplt.YTicks(fontsize=24),
        xticks=aplt.XTicks(fontsize=24),
        cmap=aplt.Cmap(norm="log", vmax=1.0, vmin=1.0),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=20),
        units=aplt.Units(in_kpc=True),
    )

    fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot)
    fit_plotter.figures_2d(normalized_residual_map=True)

"""
Making this plot for a paper? You can output it to hard disk.
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    # Only one `Analysis` so take first and only dataset.
    fit = fit_list[0]

    mat_plot = aplt.MatPlot2D(
        title=aplt.Title(label="Hey"),
        output=aplt.Output(
            path=path.join("output", "path", "of", "file"),
            filename="publication",
            format="png",
        ),
    )

"""
__Errors (Random draws from PDF)__

In the `database/models.py` example we showed how `Tracer objects could be randomly drawn form the Probability 
Distribution Function, in order to quantity things such as errors.

The same approach can be used with `FitImaging` objects, to investigate how the properties of the fit vary within
the errors (e.g. showing source reconstructions fot different fits consistent with the errors).
"""
fit_agg = al.agg.FitImagingAgg(aggregator=agg)
fit_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)


for fit_list_gen in fit_gen:  # Total samples 2 so fit_list_gen contains 2 fits.
    for fit_list in fit_list_gen:  # Iterate over each fit of total_samples=2
        # Only one `Analysis` so take first and only dataset.
        fit = fit_list[0]

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

"""
Finished.
"""
