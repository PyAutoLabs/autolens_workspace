"""
Overview: Modeling
------------------

Lens modeling is the process of taking data of a strong lens (e.g. imaging data from the Hubble Space Telescope or
interferometer data from ALMA) and fitting it with a lens model, to determine the light and mass distributions of the
lens and source galaxies that best represent the observed strong lens.

Lens modeling uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import **PyAutoFit** separately to **PyAutoLens**
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autolens as al
import autolens.plot as aplt

import autofit as af

"""
__Dataset__

In this example, we model Hubble Space Telescope imaging of a real strong lens system, with our goal to
infer the lens and source galaxy light and mass models that fit the data well!
"""
dataset_path = path.join("dataset", "slacs", "slacs1430+4105")

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We next mask the dataset, to remove the exterior regions of the image that do not contain emission from the lens or
source galaxy.

Note how when we plot the `Imaging` below, the figure now zooms into the masked region.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose the lens model that we fit to the data using `af.Model` objects. 

These behave analogously to `Galaxy` objects but their  `LightProfile` and `MassProfile` parameters are not specified,
they are instead determined by a fitting procedure.

We will fit our strong lens data with two galaxies:

- A lens galaxy with a `Sersic` bulge and  an `Isothermal` mass profile representing its mass, whose centres are 
  fixed to (0.0", 0.0").
  
- A source galaxy with an `Exponential` light profile representing a disk.

The redshifts of the lens (z=0.285) and source (z=0.575) are fixed.
"""
# Lens:
bulge = af.Model(al.lp.Sersic)
bulge.centre = (0.0, 0.0)

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)

lens = af.Model(al.Galaxy, redshift=0.285, bulge=bulge, mass=mass)

# Source:

disk = af.Model(al.lp.ExponentialCore)

source = af.Model(al.Galaxy, redshift=0.575, disk=disk)

"""
The `info` attribute of each `Model` component shows the model in a readable format.
"""
print(lens.info)
print()
print(source.info)

"""
We combine the lens and source model galaxies above into a `Collection`, which is the final lens model we will fit. 

The reason we create separate `Collection`'s for the `galaxies` and `model` is so that the `model` can be extended to 
include other components than just galaxies.
"""
galaxies = af.Collection(lens=lens, source=source)
model = af.Collection(galaxies=galaxies)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Non-linear Search__

We now choose the non-linear search, which is the fitting method used to determine the set of light and mass profile 
parameters that best-fit our data.

In this example we use `Nautilus` (https://github.com/joshspeagle/Nautilus), a nested sampling algorithm that is
very effective at lens modeling.

PyAutoLens supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
lens models at once on your CPU.
"""
search = af.Nautilus(path_prefix="overview", name="modeling", number_of_cores=4)

"""
The non-linear search fits the lens model by guessing many lens models over and over iteratively, using the models which
give a good fit to the data to guide it where to guess subsequent model. 

An animation of a non-linear search fitting another HST lens is shown below, where initial lens models give a poor 
fit to the data but gradually improve (increasing the likelihood) as more iterations are performed.

![CCD Animation](https://github.com/Jammy2211/autocti_workspace/blob/main/dataset/ccd.gif "ccd")

**Credit: Amy Etherington**

__Analysis__

We next create an `AnalysisImaging` object, which contains the `log_likelihood_function` that the non-linear search 
calls to fit the lens model to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to 
   the dataset such that a log likelihood is returned.

 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.

The log likelihood evaluation time can be estimated before a fit using the `profile_log_likelihood_function` method,
which returns two dictionaries containing the run-times and information about the fit.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

"""
The overall log likelihood evaluation time is given by the `fit_time` key.

For this example, it is ~0.01 seconds, which is extremely fast for lens modeling. More advanced lens
modeling features (e.g. shapelets, multi Gaussian expansions, pixelizations) have slower log likelihood evaluation
times (1-3 seconds), and you should be wary of this when using these features.

Feel free to go ahead a print the full `run_time_dict` and `info_dict` to see the other information they contain. The
former has a break-down of the run-time of every individual function call in the log likelihood function, whereas the 
latter stores information about the data which drives the run-time (e.g. number of image-pixels in the mask, the
shape of the PSF, etc.).
"""
print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

Estimating this quantity is more tricky, as it varies depending on the lens model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted.

For this example, we conservatively estimate that the non-linear search will perform ~10000 iterations per free 
parameter in the model. This is an upper limit, with models typically converging in far fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. Parallelization with Nautilus scales well, it speeds up the model-fit by the 
`number_of_cores` for N < 8 CPUs and roughly `0.5*number_of_cores` for N > 8 CPUs. This scaling continues 
for N> 50 CPUs, meaning that with super computing facilities you can always achieve fast run times!
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
Nautilus samples, model parameters, visualization) to hard-disk.

Once running you should checkout the `autolens_workspace/output` folder, which is where the results of the search are 
written to hard-disk on-the-fly. This includes lens model parameter estimates with errors non-linear samples and the 
visualization of the best-fit lens model inferred by the search so far. 

NOTE: This fit will take ~10 minutes to run.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Results__

Whilst navigating the output folder, you may of noted the results were contained in a folder that appears as a random
collection of characters. 

This is the model-fit's unique identifier, which is generated based on the model, search and dataset used by the fit. 
Fitting an identical model, search and dataset will generate the same identifier, meaning that rerunning the script 
will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset, a new 
unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.

The fit above returns a `Result` object, which includes lots of information on the lens model. 

The `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
Below, we print the maximum log likelihood model inferred.
"""
print(result.max_log_likelihood_instance.galaxies.lens)
print(result.max_log_likelihood_instance.galaxies.source)

"""
The result contains the full posterior information of our non-linear search, including all parameter samples, 
log likelihood values and tools to compute the errors on the lens model. 

There are built in visualization tools for plotting this.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
The result also contains the maximum log likelihood `Tracer` and `FitImaging` objects which can easily be plotted.

The fit has more significant residuals than the previous tutorial. It is clear that the lens model cannot fully
capture the central emission of the lens galaxy and the complex structure of the lensed source galaxy. Nevertheless, 
it is sufficient to estimate simple lens quantities, like the Einstein Mass.

The next examples cover all the features that **PyAutoLens** has to improve the model-fit.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=dataset.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
A full guide of result objects is contained in the `autolens_workspace/*/imaging/results` package.

__Model Customization__

The model can be fully customized, making it simple to parameterize and fit many different lens models
using any combination of light and mass profiles.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.DevVaucouleurs, mass=al.mp.Isothermal
)

"""
This aligns the light and mass profile centres in the model, reducing the
number of free parameter fitted for by Nautilus by 2.
"""
lens.bulge.centre = lens.mass.centre

"""
This fixes the lens galaxy light profile's effective radius to a value of
0.8 arc-seconds, removing another free parameter.
"""
lens.bulge.effective_radius = 0.8

"""
This forces the mass profile's einstein radius to be above 1.0 arc-seconds.
"""
lens.mass.add_assertion(lens.mass.einstein_radius > 1.0)

"""
The `info` attribute shows the customized model.
"""
print(lens.info)

"""
__Cookbook__

The readthedocs contain a modeling cookbook which provides a concise reference to all the ways to customize a lens 
model: https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Linear Light Profiles__

**PyAutoLens** supports 'linear light profiles', where the `intensity` parameters of all parametric components are 
solved via linear algebra every time the model is fitted using a process called an inversion. This inversion always 
computes `intensity` values that give the best fit to the data (e.g. they maximize the likelihood) given the other 
parameter values of the light profile.

The `intensity` parameter of each light profile is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in the example below by 3) and removing 
the degeneracies that occur between the `intensity` and other light profile
parameters (e.g. `effective_radius`, `sersic_index`).

For complex models, linear light profiles are a powerful way to simplify the parameter space to ensure the best-fit
model is inferred.

A full descriptions of this feature is given in the `linear_light_profiles` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/linear_light_profiles.ipynb

__Multi Gaussian Expansion__

A natural extension of linear light profiles are basis functions, which group many linear light profiles together in
order to capture complex and irregular structures in a galaxy's emission.

Using a clever model parameterization a basis can be composed which corresponds to just N = 4-6 parameters, making
model-fitting efficient and robust.

A full descriptions of this feature is given in the `multi_gaussian_expansion` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/multi_gaussian_expansion.ipynb

__Shapelets__

**PyAutoLens** also supports `Shapelets`, which are a powerful way to fit the light of the galaxies which
typically act as the source galaxy in strong lensing systems.

A full descriptions of this feature is given in the `shapelets` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/shapelets.ipynb

__Pixelizations__

The source galaxy can be reconstructed using adaptive pixel-grids (e.g. a Voronoi mesh or Delaunay triangulation), 
which unlike light profiles, a multi Gaussian expansion or shapelets are not analytic functions that conform to 
certain symmetric profiles. 

This means they can reconstruct more complex source morphologies and are better suited to performing detailed analyses
of a lens galaxy's mass.

A full descriptions of this feature is given in the `pixelization` example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/features/pixelization.ipynb

The fifth overview example of the readthedocs also give a description of pixelizations:

https://pyautolens.readthedocs.io/en/latest/overview/overview_5_pixelizations.html

__Wrap Up__

A more detailed description of lens modeling is provided at the following example:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/imaging/modeling/start_here.ipynb

Chapters 2 and 3 **HowToLens** lecture series give a comprehensive description of lens modeling, including a
description of what a non-linear search is and strategies to fit complex lens model to data in efficient and
robust ways.
"""
