"""
Results: Galaxies and Fits
===========================

This tutorial inspects an inferred model using galaxies inferred by the non-linear search.
This allows us to visualize and interpret its results.

The galaxies and fit API is described fully in the guides:

 - `autolens_workspace/*/guides/tracer.ipynb`
 - `autolens_workspace/*/guides/fit.ipynb`
 - `autolens_workspace/*/guides/galaxies.ipynb`

This result example only explains specific functionality for using a `Result` object to inspect galaxies or a fit
and therefore you should read these guides in detail first.

__Plot Module__

This example uses the plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autolens_workspace/*/plot` package in full.
This includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoLens**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.

The guide `guides/units_and_cosmology.ipynb` illustrates how to convert these quantities to physical units like
kiloparsecs, magnitudes and solar masses.

__Data Structures__

Quantities inspected in this example script use **PyAutoLens** bespoke data structures for storing arrays, grids,
vectors and other 1D and 2D quantities. These use the `slim` and `native` API to toggle between representing the
data in 1D numpy arrays or high dimension numpy arrays.

This tutorial will only use the `slim` properties which show results in 1D numpy arrays of
shape [total_unmasked_pixels]. This is a slimmed-down representation of the data in 1D that contains only the
unmasked data points

These are documented fully in the `autolens_workspace/*/guides/data_structures.ipynb` guide.

__Start Here Notebook__

If any code in this script is unclear, refer to the `results/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Model Fit__

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore, disk=None
        ),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = al.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)


"""
__Max Likelihood Tracer__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_tracer` which we can visualize.
"""
tracer = result.max_log_likelihood_tracer

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=mask.derive_grid.all_false)
tracer_plotter.subplot_tracer()

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the tracer changes for models with similar log likelihoods. Below, we create and plot
the tracer of the 100th last accepted model by Nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

# Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
tracer = al.Tracer(galaxies=instance.galaxies)
fit = al.FitImaging(dataset=dataset, tracer=tracer)
tracer = fit.tracer_linear_light_profiles_to_light_profiles

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=mask.derive_grid.all_false)
tracer_plotter.subplot_tracer()

"""
__Samples API__

In the first results tutorial, we used `Samples` objects to inspect the results of a model.

We saw how these samples created instances, which include a `galaxies` property that mains the API of the `Model`
creates above (e.g. `galaxies.source.bulge`). 

We can also use this instance to extract individual components of the model.
"""
samples = result.samples

ml_instance = samples.max_log_likelihood()

# Input to FitImaging to solve for linear light profile intensities, see `start_here.py` for details.
tracer = al.Tracer(galaxies=instance.galaxies)
fit = al.FitImaging(dataset=dataset, tracer=tracer)
tracer = fit.tracer_linear_light_profiles_to_light_profiles

bulge = tracer.galaxies.source.bulge

bulge_image_2d = bulge.image_2d_from(grid=dataset.grid)
print(bulge_image_2d.slim[0])

bulge_plotter = aplt.LightProfilePlotter(light_profile=bulge, grid=dataset.grid)
bulge_plotter.figures_2d(image=True)

"""
In fact, if we create a `Tracer` from an instance (which is how `result.max_log_likelihood_tracer` is created) we
can choose whether to access its attributes using each API: 
"""
tracer = result.max_log_likelihood_tracer
print(tracer.galaxies.source.bulge)

"""
__Max Likelihood Fit__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit` which we can visualize.
"""
fit = result.max_log_likelihood_fit

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by Nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

tracer = al.Tracer(galaxies=instance.galaxies)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Errors__

Using a `GalaxyPDFPlotter`, we can make 1D plots that show the errors of the light and mass models estimated via a 
model-fit. 

Here, the `galaxy_pdf_list` is a list of `Galaxy` objects that are drawn randomly from the PDF of the model-fit. 

These are used to estimate the errors at an input `sigma` value of: 

 - The 1D light or mass profile, which is plotted as a shaded region on the figure. 
 - The median `half_light_radius` and `einstein_radius` with errors, which are plotted as vertical lines.

Below, we manually input one hundred realisations of the lens galaxy with light and mass profiles that clearly show 
these errors on the figure.
"""
galaxy_pdf_list = [samples.draw_randomly_via_pdf().galaxies.lens for i in range(10)]

galaxy_pdf_plotter = aplt.GalaxyPDFPlotter(
    galaxy_pdf_list=galaxy_pdf_list, grid=dataset.grid, sigma=3.0
)
galaxy_pdf_plotter.figures_1d(
    #    image=True,
    #   convergence=True,
    #   potential=True
)

"""
A decomposed plot of the individual light profiles of the galaxy, with errors, can also be created.
"""
galaxy_pdf_plotter.figures_1d_decomposed(
    # image=True,
    #  convergence=True,
    #  potential=True
)

"""
__Wrap Up__

We have learnt how to extract individual planes, galaxies, light and mass profiles from the tracer that results from
a model-fit and use these objects to compute specific quantities of each component.
"""
