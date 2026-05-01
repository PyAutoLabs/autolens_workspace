"""
Results: Start Here
===================

After a lens model-fit completes, nearly everything a user could need is written to the `output/` folder. Most of it
can be loaded back into full Python objects with a single line of code, via either `.json` files (for model objects
like the `Tracer`, `Model` and samples) or `.fits` files (for imaging products like the model image, residuals and
source-plane images).

This guide shows two complementary ways to get those results back into Python:

  - **Simple loading** — point at a single fit's `output/...` folder and load `.json` / `.fits` files directly with
    `from_json(...)` and `al.Imaging.from_fits(...)`. The objects that come back behave exactly like the in-memory
    `Result` returned by `search.fit()`. This is the fastest way to inspect one fit, but everything you load sits
    in memory.
  - **Aggregator** — scrape a directory of completed fits and yield the same objects (`Tracer`, `Samples`, `Model`,
    ...) via Python generators, so you can iterate over hundreds of fits without holding them all in memory at
    once. This is the right tool when you want to analyse a sample of lenses together.

Both sections appear below in that order. The aggregator section first runs a quick model-fit so a results
directory exists, then walks through the deeper API (samples, fits, tracer, units, pixelization). Where each
aggregator section reaches a result that the simple-loading API also exposes, this is noted — both routes return
the same PyAutoFit / PyAutoLens objects, just sourced from disk in different ways.

__Output Folder Layout__

Each completed fit lives at a path like::

    output/imaging/<dataset_name>/modeling/<unique_hash>/
        files/                     <- JSON + CSV: loadable Python objects
            tracer.json            <- max log likelihood Tracer
            model.json             <- fitted af.Collection model
            samples.csv            <- full Nautilus samples
            samples_summary.json   <- max log likelihood parameter values + errors
            samples_info.json      <- metadata about the samples
            search.json            <- non-linear search configuration
            settings.json          <- search settings
            cosmology.json         <- cosmology used for the fit
            covariance.csv         <- parameter covariance matrix
        image/                     <- FITS: imaging products
            dataset.fits           <- data, noise-map and PSF
            fit.fits               <- model image, residuals, chi-squared map
            tracer.fits            <- tracer image-plane images per galaxy
            source_plane_images.fits  <- source plane reconstructions
            model_galaxy_images.fits  <- per-galaxy model images
            galaxy_images.fits        <- per-galaxy images
            dataset.png, fit.png, tracer.png   <- visualisations
        model.info                 <- human-readable model summary
        model.results              <- human-readable fit summary
        search.summary             <- search run summary
        metadata                   <- run metadata

__Contents__

**Simple Loading (one fit at a time):**

**Tracer:** Load the maximum log likelihood `Tracer` from `tracer.json`.
**Model:** Load the fitted `af.Collection` model from `model.json`.
**Samples:** Load the non-linear search samples from `samples.csv` / `samples_summary.json`.
**FITS Files:** Load imaging products (data, fit, tracer images) from the `image/` folder.

**Aggregator (many fits, generator-based):**

**Model Fit:** Run a quick fit so a results directory exists for the aggregator examples.
**Info:** Print the result in a readable format.
**Loading From Hard-disk:** Use `Aggregator.from_directory(...)` to scrape `output/`.
**Generators:** How Python generators give the aggregator its memory efficiency.
**Database File:** Loading from a `.sqlite` database for very large samples.
**Workflow Examples:** Building scientific workflows (CSV / PNG / FITS makers).
**Result:** Working with the in-memory `Result` returned by `search.fit()`.
**Samples:** Median-PDF model and parameter errors from the `Samples` object.
**Linear Light Profiles:** Reading `intensity` values solved by linear algebra.
**Tracer:** Producing images and lensing quantities from the maximum likelihood `Tracer`.
**Fits:** Inspecting the `FitImaging` object (chi-squared, log likelihood).
**Galaxies:** Accessing individual lens / source galaxies inside the `Tracer`.
**Units and Cosmological Quantities:** Converting parameters to physical units.
**Linear Light Profiles / Basis Objects:** Specific functionality for linear light profiles and basis functions.
**Pixelization:** Pixelized source reconstructions on a Voronoi mesh.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports
from autoconf.dictable import from_json

# from autoconf import setup_notebook; setup_notebook()

import os
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
==============================================================================
                              SIMPLE LOADING
==============================================================================

The first half of this guide loads a single fit directly from `output/`. This is the fastest way to inspect one
fit — every file under `files/` and `image/` is a Python object away.

__Result Path__

Set the path to the folder of the fit you want to load. Replace the values below with the path to your own fit.

The layout below assumes the canonical `output/imaging/<dataset>/modeling/<hash>/` structure produced by
`scripts/imaging/modeling.py`. The `<unique_hash>` placeholder is a 32-character identifier specific to the fit;
each load below is guarded with `.exists()` so this script runs cleanly even before you replace it.
"""
result_path = (
    Path("output")
    / "imaging"
    / "simple"
    / "modeling"
    / "<unique_hash>"  # The 32-character identifier for the specific fit.
)

files_path = result_path / "files"
image_path = result_path / "image"

"""
__Tracer__

The maximum log likelihood `Tracer` is saved to `files/tracer.json` and can be loaded in one line.

The `Tracer` contains every `Galaxy`, light profile and mass profile at their max log likelihood values, so it can
be used directly to compute convergence maps, deflection angles, source-plane images and more — exactly as if it
had been returned by `search.fit()`.
"""
if (files_path / "tracer.json").exists():
    tracer = from_json(file_path=files_path / "tracer.json")

    print(tracer)
    print(tracer.galaxies)

"""
__Model__

The fitted `af.Collection` model is saved to `files/model.json`. This is the *prior* model (with free parameters),
not the max log likelihood instance — useful for inspecting the structure of what was fitted.
"""
if (files_path / "model.json").exists():
    model = from_json(file_path=files_path / "model.json")
    print(model.info)

"""
__Samples__

The full set of non-linear search samples is saved to `files/samples.csv` and its summary to
`files/samples_summary.json`. Both can be loaded without re-running the search.
"""
if (files_path / "samples.csv").exists() and (files_path / "model.json").exists():
    model = from_json(file_path=files_path / "model.json")
    samples = af.SamplesNest.from_csv(file_path=files_path / "samples.csv", model=model)
    print(samples.max_log_likelihood())

"""
__FITS Files__

The `image/` folder contains the imaging products of the fit as `.fits` files. These load with the standard
`al.Imaging` / `al.Array2D` APIs and can be plotted with `aplt`.
"""
if (image_path / "dataset.fits").exists():
    dataset = al.Imaging.from_fits(
        data_path=image_path / "dataset.fits",
        noise_map_path=image_path / "dataset.fits",
        psf_path=image_path / "dataset.fits",
        data_hdu=0,
        noise_map_hdu=1,
        psf_hdu=2,
        pixel_scales=0.1,
    )

if (image_path / "tracer.fits").exists():
    tracer_images = al.Array2D.from_fits(
        file_path=image_path / "tracer.fits", hdu=0, pixel_scales=0.1
    )

"""
==============================================================================
                                AGGREGATOR
==============================================================================

Everything above loaded one fit at a time, by pointing at a specific output directory. To analyse many fits — for
example a sample of hundreds of lenses — use the **aggregator** instead. It scrapes a directory of fits and yields
the same objects (`Tracer`, `Samples`, `Model`, ...) via Python generators, keeping memory use low.

The remainder of this file is the aggregator entry point. After reading it, the sibling files in `aggregator/`
provide more detailed examples for analysing different aspects of the results.

To make the examples below runnable from a fresh checkout, we first perform a quick model-fit so the aggregator
has a results directory to scrape. Anything `from_json(...)` reaches in the simple-loading section above can also
be reached through the aggregator below — both APIs return the same Python objects.

If you are not familiar with the lens modeling API, see `autolens_workspace/*/examples/modeling/` first.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/features/no_lens_light/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=bulge, disk=None),
    ),
)

test_mode_was_on = os.environ.get("PYAUTO_TEST_MODE") == "1"
if test_mode_was_on:
    os.environ.pop("PYAUTO_TEST_MODE", None)

search = af.Nautilus(
    path_prefix=Path("results_folder"),
    name="results",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in `modeling` examples.
    **({"n_like_max": 300} if test_mode_was_on else {}),
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

result = search.fit(model=model, analysis=analysis)

if test_mode_was_on:
    os.environ["PYAUTO_TEST_MODE"] = "1"

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
for example the `model` is loaded as the same `Model` object the simple-loading section above reached via
`from_json(file_path="files/model.json")`.

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
 - `settings`: The settings associated with a inversion if used (`settings.json`).
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
    directory=Path("output") / "results_folder",
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

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

It is recommended you use hard-disk loading to begin, as it is simpler and easier to use.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.

__Workflow Examples__

The `results/workflow` folder contains examples describing how to build a scientific workflow using the results
of model-fits, in order to quickly and easily inspect and interpret results.

These examples use functionality designed for modeling large dataset samples, with the following examples:

- `csv_maker.py`: Make .csv files from the modeling results which summarize the results of a large sample of fits.
- `png_maker.py`: Make .png files of every fit, to quickly check the quality of the fit and interpret the results.
- `fits_maker.py`: Make .fits files of every fit, to quickly check the quality of the fit and interpret the results.

The above examples work on the raw outputs of the model-fits that are stored in the `output` folder, for example
the visualization .png files, the .fits files containing results and parameter inferences which make the .csv files.

They are therefore often quick to run and allow you to make a large number of checks on the results of your model-fits
in a short period of time.

Below is a quick example, where we use code from the `csv_maker.py` scripts to create a .csv file from the fit above,
containing the inferred Einstein radius, in a folder you can inspect quickly.

The `workflow_path` specifies where these files are output, in this case the .csv files which summarise the results,
and the code below can easily be adapted to output the .png and .fits files.
"""
workflow_path = Path("output") / "results_folder_csv_png_fits" / "workflow_make_example"

agg_csv = af.AggregateCSV(aggregator=agg)
agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius"
)  # Example of adding a column
agg_csv.save(path=workflow_path / "csv_very_simple.csv")

"""
__Result__

From here on we will use attributes contained in the `result` passed from the `search.fit` method above, as opposed
to using the aggregator. This is because things will run faster, but all of the results we use can be loaded using
the aggregator as shown above (or via the simple-loading API in the first half of this file).

__Samples__

The result's `Samples` object contains the complete set of non-linear search Nautilus samples, where each sample
corresponds to a set of a model parameters that were evaluated and accepted. This is the same `Samples` object that
`af.SamplesNest.from_csv(file_path="files/samples.csv", model=model)` returned in the simple-loading section.

The examples script `autolens_workspace/*/guides/results/aggregator/samples.py` provides a detailed description of
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
__Linear Light Profiles__

In the model fit, linear light profiles are used, solving for the `intensity` of each profile through linear algebra.

The `intensity` value is not a free parameter of the linear light profiles in the model, meaning that in the `Samples`
object the `intensity` are always defaulted to values of 1.0 in the `Samples` object.

You can observe this by comparing the `intensity` values in the `Samples` object to those in
the `result.max_log_likelihood_galaxies` instance and `result.max_log_likelihood_fit` instance.
"""
samples = result.samples
ml_instance = samples.max_log_likelihood()

print(
    "Intensity of source galaxy's bulge in the Samples object (before solving linear algebra):"
)
print(ml_instance.galaxies.source.bulge.intensity)

print(
    "Intensity of source galaxy's bulge in the max log likelihood galaxy (after solving linear algebra):"
)
print(result.max_log_likelihood_tracer.planes[1][0].bulge.intensity)
print(
    result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles.planes[
        1
    ][0].bulge.intensity
)

"""
To interpret results associated with the linear light profiles, you must input the `Samples` object into a `FitImaging`,
which converts the linear light profiles to standard light profiles with `intensity` values solved for using the linear
algebra.
"""
ml_instance = samples.max_log_likelihood()

tracer = al.Tracer(galaxies=ml_instance.galaxies)
fit = al.FitImaging(dataset=dataset, tracer=tracer)
tracer = fit.tracer_linear_light_profiles_to_light_profiles

print("Intensity of source galaxy's bulge after conversion using FitImaging:")
print(tracer.planes[1][0].bulge.intensity)

"""
Whenever possible, the result already containing the solved `intensity` values is used, for example
the `Result` object returned by a search.

However, when manually loading results from the `Samples` object, you must use the `FitImaging` object to convert
the linear light profiles to their correct `intensity` values.

__Tracer__

The result's maximum likelihood `Tracer` object contains everything necessary to perform ray-tracing and other
calculations with the lens model. It is the same `Tracer` that the simple-loading section above reached via
`from_json(file_path="files/tracer.json")`.

The guide `autolens_workspace/*/guides/tracer.py` provides a detailed description of this object, including:

 - Producing individual images of the strong lens from a tracer.
 - Inspecting mass model components like the convergence, potential and deflection angles.
 - Other lensing quantities like the critical curve and caustics.

The examples script `autolens_workspace/*/guides/results/aggregator/galaxies_fit.py` show how to use
model-fitting results specific functionality of galaxies, including:

 - Drawing tracers from the samples and plotting their images.
 - Producing 1D plots of the galaxy's light and mass profiles with error bars.

Below, is an example of how to use the `Tracer` object to calculate the image of the lens and source galaxies.
"""
tracer = result.max_log_likelihood_tracer

image = tracer.image_2d_from(grid=dataset.grid)

"""
__Fits__

The result's maximum likelihood `FitImaging` object contains everything necessary to inspect the lens model fit to the
data.

The guide `autolens_workspace/*/guides/fits.py` provides a detailed description of this object, including:

 - Performing a fit to data with galaxies.
 - Inspecting the model data, residual-map, chi-squared, noise-map of the fit.
 - Other properties of the fit that inspect how good it is.

The examples script `autolens_workspace/*/guides/results/aggregator/galaxies_fits.py` provides a detailed description of this
object, including:

 - Repeating fits using the results contained in the samples.

Below, is an example of how to use the `FitImaging` object to print the maximum likelihood chi-squared and
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

The examples script `autolens_workspace/*/guides/results/aggregator/galaxies_fits.py` shows how to use
model-fitting results specific functionality of galaxies, including:

 - Repeating fits using the results contained in the samples.

Below, is an example of how to use the `Galaxy` objects to plot the source galaxy's source-plane image.
"""
tracer = result.max_log_likelihood_tracer

source = tracer.planes[1][0]

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

Below, is an example of how to convert the y centre of the source galaxy from arcseconds to kiloparsecs.
"""
tracer = result.max_log_likelihood_tracer

cosmology = al.cosmo.Planck15()

source = tracer.planes[1][0]
source_kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=source.redshift)
source_centre_0_kpc = source.bulge.centre[0] * source_kpc_per_arcsec

"""
__Linear Light Profiles / Basis Objects__

A lens model can be fitted using a linear light profile, which is a light profile whose `intensity` parameter is
sovled for via linear algebra.

This includes Basis objects such as a Multi-Gaussian expansion of Shapelets.

These objects mostly behave identically to ordinary light profiles, but due to the linear algebra have their own
specific functionality.

The example script `autolens_workspace/*/features/linear_light_profiles.py` provides a detailed description of
using linear light profile results including:

 - Extracting individual quantities from the linear light profile, such as the coefficients of the basis functions.
 - Extracting the intensity of the linear light profiles after they have been computed via linear algebra.
 - Plotting the linear light profiles.

Therefore if your results contain a linear light profile, checkout the example script above for a detailed description
of how to use their results.

__Pixelization__

The lens model can reconstruct the source galaxy using a pixelization, for example on a Voronoi mesh.

The example script `autolens_workspace/*/features/pixelization.py` describes using pixelization results including:

 - Producing source reconstructions using the Voronoi mesh, RectangularAdaptDensity triangulation or whichever mesh is used.
 - Inspecting the evidence terms of the fit, which quantify how well the pixelization reconstructs fits the data whilst
   accounting for the complexity of the pixelization.
 - Estimating the magnification of the source galaxy's image using the pixelization.

Therefore if your results contain a pixelization, checkout the example script above for a detailed description
of how to use their results.
"""
