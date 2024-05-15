"""
Results: Start Here
===================

This script is the starting point for investigating the results of lens modeling and it provides
an overview of the lens modeling API.

The majority of results are dataset independent, meaning that the same API can be used to inspect the results of any
lens model. Therefore, for the majority of results we refer you to the `autolens_workspace/imaging/results` package,
which details the API which can be copy and pasted for interferometer fits.

The `examples` folder here does provide specific examples of how to inspects the results of fits using
interferometer datasets.

__Model__

We begin by fitting a quick lens model to a simple lens dataset, which we will use to illustrate the lens modeling
results API.

If you are not familiar with the lens modeling API and process, checkout the `autolens_workspace/examples/modeling`
folder for examples.
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
__Model Fit__

The code below (which we have omitted comments from for brevity) performs a lens model-fit using Nautilus. You should
be familiar enough with lens modeling to understand this, if not you should go over the beginner model-fit script again!
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=4.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

bulge = af.Model(al.lp.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=path.join("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = al.AnalysisInterferometer(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result.info)


"""
__Loading From Hard-disk__

When performing fits which output results to hard-disk, a `files` folder is created containing .json / .csv files of 
the model, samples, search, etc. You should check it out now for a completed fit on your hard-disk if you have
not already!

These files can be loaded from hard-disk to Python variables via the aggregator, making them accessible in a 
Python script or Jupyter notebook. They are loaded as the internal **PyAutoFit** objects we are familiar with,
for example the `model` is loaded as the `Model` object we passed to the search above.

Below, we will access these results using the aggregator's `values` method. A full list of what can be loaded is
as follows:

 - `model`: The `model` defined above and used in the model-fit (`model.json`).
 - `search`: The non-linear search settings (`search.json`).
 - `samples`: The non-linear search samples (`samples.csv`).
 - `samples_info`: Additional information about the samples (`samples_info.json`).
 - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
 - `info`: The info dictionary passed to the search (`info.json`).
 - `covariance`: The inferred covariance matrix (`covariance.csv`).
 - `cosmology`: The cosmology used by the fit (`cosmology.json`).
 - `settings_inversion`: The settings associated with a inversion if used (`settings_inversion.json`).
 - `dataset/data`: The data that is fitted (`data.fits`).
 - `dataset/noise_map`: The noise-map (`noise_map.fits`).
 - `dataset/psf`: The Point Spread Function (`psf.fits`).
 - `dataset/mask`: The mask applied to the data (`mask.fits`).
 - `dataset/settings`: The settings associated with the dataset (`settings.json`).

The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does not reperform calculations using the full 
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the `samples.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder"),
)

"""
__Generators__

Before using the aggregator to inspect results, lets discuss Python generators. 

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects 
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory 
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once. 
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the 
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the 
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the `values` method. This takes the `name` of the
object we want to create a generator of, for example inputting `name=samples` will return the results `Samples`
object (which is illustrated in detail below).
"""
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

"""
__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is benefitial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

It is recommended you use hard-disk loading to begin, as it is simpler and easier to use.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.

__Result__

From here on we will use attributes contained in the `result` passed from the `search.fit` method above, as opposed
to using the aggregator. This is because things will run faster, but all of the results we use can be loaded using
the aggregator as shown above.

__Samples__

The result's `Samples` object contains the complete set of non-linear search Nautilus samples, where each sample 
corresponds to a set of a model parameters that were evaluated and accepted. 

The examples script `autolens_workspace/*/imaging/results/examples/samples.py` provides a detailed description of 
this object, including:

 - Extracting the maximum likelihood lens model.
 - Using marginalized PDFs to estimate errors on the lens model parameters.
 - Deriving errors on derived quantities, such as the Einstein radius.

Below, is an example of how to use the `Samples` object to estimate the lens mass model parameters which are 
the median of the probability distribution function and its errors at 3 sigma confidence intervals.
"""
samples = result.samples

median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance.galaxies.lens.mass)
print()

ue3_instance = samples.values_at_upper_sigma(sigma=3.0)
le3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance.galaxies.lens.mass, "\n")
print(le3_instance.galaxies.lens.mass, "\n")

"""
__Tracer__

The result's maximum likelihood `Tracer` object contains everything necessary to perform ray-tracing and other
calculations with the lens model.

The guide `autolens_workspace/*/guides/tracer.py` provides a detailed description of this object, including:

 - Producing individual images of the strong lens from a tracer.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

The examples script `autolens_workspace/*/imaging/results/examples/galaxies_fit.py` show how to use 
model-fitting results specific functionality of galaxies, including:

 - Drawing tracers from the samples and plotting their images.
 - Producing 1D plots of the galaxy's light and mass profiles with error bars.

Below, is an example of how to use the `Tracer` object to calculate the image of the lens and source galaxies.
"""
tracer = result.max_log_likelihood_tracer

image = tracer.image_2d_from(grid=dataset.grid)

"""
__Fits__

The result's maximum likelihood `FitInterferometer` object contains everything necessary to inspect the lens model 
The result's maximum likelihood `FitImaging` object contains everything necessary to inspect the lens model fit to the 
data.

The guide `autolens_workspace/*/guides/fits.py` provides a detailed description of this object, including:

 - Performing a fit to data with galaxies.
 - Inspecting the model data, residual-map, chi-squared, noise-map of the fit.
 - Other properties of the fit that inspect how good it is.

The examples script `autolens_workspace/*/imaging/results/examples/galaxies_fits.py` provides a detailed description of this 
object, including:

 - Repeating fits using the results contained in the samples.

This script uses a `FitImaging` object, but the API for the majority of quantities are identical for an 
interferometer fit.

Below, is an example of how to use the `FitInterferometer` object to print the maximum likelihood chi-squared and 
log likelihood values.
"""
fit = result.max_log_likelihood_fit

print(fit.chi_squared)
print(fit.log_likelihood)

"""
__Galaxies__

The result's maximum likelihood `Galaxy` objects contained within the `Tracer` contain everything necessary to 
inspect the individual properties of the lens and source galaxies.

The guide `autolens_workspace/*/guides/fits.py` provides a detailed description of this, including:

 - Extracting the lens and source galaixes from a tracer.
 - Extracting the individual light and mass profiles of the galaxies.

The examples script `autolens_workspace/*/imaging/results/examples/galaxies_fits.py` shows how to use 
model-fitting results specific functionality of galaxies, including:

 - Repeating fits using the results contained in the samples.
 
Below, is an example of how to use the `Galaxy` objects to plot the source galaxy's source-plane image.
"""
tracer = result.max_log_likelihood_tracer

source = tracer.planes[1][0]
galaxy_plotter = aplt.GalaxyPlotter(galaxy=source, grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Units and Cosmological Quantities__

The maximum likelihood model includes cosmological quantities, which can be computed via the result.

The examples script `autolens_workspace/*/guides/units_and_cosmology.py` provides a detailed 
description of this object, including:

 - Calculating the Einstein radius of the lens galaxy.
 - Converting quantities like the Einstein radius or effective radius from arcseconds to kiloparsecs.
 - Computing the Einstein mass of the lens galaxy in solar masses.
 
This guide is not in the `results` package but the `guides` package, as it is a general guide to the
**PyAutoLens** API. However, it may be useful when inspecting results.
 
Below, is an example of how to convert the effective radius of the source galaxy from arcseconds to kiloparsecs.
"""
tracer = result.max_log_likelihood_tracer

cosmology = al.cosmo.Planck15()

source = tracer.planes[1][0]
source_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=source.redshift)
source_effective_radius_kpc = source.bulge.effective_radius * source_kpc_per_arcsec

"""
__Linear Light Profiles / Basis Objects__

A lens model can be fitted using a linear light profile, which is a light profile whose `intensity` parameter is 
sovled for via linear algebra.

This includes Basis objects such as a Multi-Gaussian expansion of Shapelets.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality.

The example script `autolens_workspace/*/imaging/modeling/linear_light_profiles.py` provides a detailed description of 
using linear light profile results including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.
 
Therefore if your results contain a linear light profile, checkout the example script above for a detailed description
of how to use their results.

__Pixelization__

The lens model can reconstruct the source galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autolens_workspace/*/imaging/modeling/pixelization.py` describes using pixelization results including:

 - Producing source reconstructions using the Voronoi mesh, Delaunay triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the source galaxy's image using the pixelization.

Therefore if your results contain a pixelization, checkout the example script above for a detailed description
of how to use their results.
"""
