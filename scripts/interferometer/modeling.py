"""
Modeling: Start Here
====================

This script is the starting point for lens modeling of interferometer datasets (e.g. ALMA, VLBI) with
**PyAutoLens** and it provides an overview of the lens modeling API.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing lens
modeling in different ways and customizing the analysis.

__Model__

This script fits `Interferometer` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric `SersicCore`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the strong lens is evaluated using.
"""
mask_radius = 4.0

real_space_mask = al.Mask2D.circular(
    shape_native=(800, 800), pixel_scales=0.05, radius=mask_radius
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files, which we will fit 
with the lens model.

This includes the method used to Fourier transform the real-space image of the strong lens to the uv-plane and compare 
directly to the visiblities. We use a non-uniform fast Fourier transform, which is the most efficient method for 
interferometer datasets containing ~1-10 million visibilities.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Over Sampling__

If you are familiar with using imaging data, you may have seen that a numerical technique called over sampling is used, 
which evaluates light profiles on a higher resolution grid than the image data to ensure the calculation is accurate.

Interferometer does not observe galaxies in a way where over sampling is necessary, therefore all interferometer
calculations are performed without over sampling.

__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - An `Sersic` `LightProfile` for the source galaxy's light [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

__Linear Light Profiles__

The model below uses a `linear light profile` for the bulge and disk, via the API `lp_linear`. This is a specific type 
of light profile that solves for the `intensity` of each profile that best fits the data via a linear inversion. 
This means it is not a free parameter, reducing the dimensionality of non-linear parameter space. 

Linear light profiles significantly improve the speed, accuracy and reliability of modeling and they are used
by default in every modeling example. A full description of linear light profiles is provided in the
`autolens_workspace/*/modeling/features/linear_light_profiles.py` example.

A standard light profile can be used if you change the `lp_linear` to `lp`, but it is not recommended.

__Coordinates__

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/guides/modeling/customize`).
"""
# Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=10,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.

[The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).]
"""
print(model.info)

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

The folders: 

 - `autolens_workspace/*/guides/modeling/searches`.
 - `autolens_workspace/*/guides/modeling/customize`
  
Give overviews of the non-linear searches **PyAutoLens** supports and more details on how to customize the
model-fit, including the priors on the model.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/imaging/simple/mass[sie]_source[bulge]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.
"""
search = af.Nautilus(
    path_prefix=Path("interferometer") / "modeling",
    name="start_here",
    unique_tag=dataset_name,
    n_live=75,
    iterations_per_quick_update=50000,
)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` used by the non-linear search to fit the 
model to the `Interferometer`dataset.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    settings_inversion=al.SettingsInversion(use_w_tilde=False),
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to 
   the dataset such that a log likelihood is returned.
 
 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.
   
For this analysis, the log likelihood evaluation time is ~0.01 seconds on CPU, < 0.001 seconds on GPU, which is 
extremely fast for lens modeling. 

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. For this model, this is typically around
? iterations, meaning that this script takes ? on CPU and ? on GPU.

__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs the 
Nautilus non-linear search in order to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Output Folder__

Now this is running you should checkout the `autolens_workspace/output` folder. This is where the results of the 
search are written to hard-disk (in the `start_here` folder), where all outputs are human readable (e.g. as .json,
.csv or text files).

As the fit progresses, results are written to the `output` folder on the fly using the highest likelihood model found
by the non-linear search so far. This means you can inspect the results of the model-fit as it runs, without having to
wait for the non-linear search to terminate.
 
The `output` folder includes:

 - `model.info`: Summarizes the lens model, its parameters and their priors discussed in the next tutorial.
 
 - `model.results`: Summarizes the highest likelihood lens model inferred so far including errors.
 
 - `images`: Visualization of the highest likelihood model-fit to the dataset, (e.g. a fit subplot showing the lens 
 and source galaxies, model data and residuals).
 
 - `files`: A folder containing .fits files of the dataset, the model as a human-readable .json file, 
 a `.csv` table of every non-linear search sample and other files containing information about the model-fit.
 
 - search.summary: A file providing summary statistics on the performance of the non-linear search.
 
 - `search_internal`: Internal files of the non-linear search (in this case Nautilus) used for resuming the fit and
  visualizing the search.

__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer,
    grid=real_space_mask.derive_grid.unmasked,
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

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
This script gives a concise overview of the basic modeling API, fitting one the simplest lens models possible.

Lets now consider what features you should read about to improve your lens modeling, especially if you are aiming
to fit more complex models to your data.

__Features__

The examples in the `autolens_workspace/*/interferometer/modeling/features` package illustrate other lens modeling 
features. 

We recommend you checkout the following two features, because they make lens modeling of interferometer datasets 
in general more reliable and  efficient (you will therefore benefit from using these features irrespective of the 
quality of your data and scientific topic of study).

We recommend you now checkout the following two features for interferometer modeling:

- ``linear_light_profiles``: The model light profiles use linear algebra to solve for their intensity, reducing model complexity.
- ``pixelization``: The source is reconstructed using an adaptive RectangularMagnification or Voronoi mesh.

The files `autolens_workspace/*/guides/modeling/searches` and `autolens_workspace/*/guides/modeling/customize`
provide guides on how to customize many other aspects of the model-fit. Check them out to see if anything
sounds useful, but for most users you can get by without using these forms of customization!
  
__Data Preparation__

If you are looking to fit your own interferometer data of a strong lens, checkout  
the `autolens_workspace/*/interferometer/data_preparation/start_here.ipynb` script for an overview of how data should be 
prepared before being modeled.

__HowToLens__

This `start_here.py` script, and the features examples above, do not explain many details of how lens modeling is 
performed, for example:

 - How does PyAutoLens perform ray-tracing and lensing calculations in order to fit a lens model?
 - How is a lens model fitted to data? What quantifies the goodness of fit (e.g. how is a log likelihood computed?).
 - How does Nautilus find the highest likelihood lens models? What exactly is a "non-linear search"?

You do not need to be able to answer these questions in order to fit lens models with PyAutoLens and do science.
However, having a deeper understanding of how it all works is both interesting and will benefit you as a scientist

This deeper insight is offered by the **HowToLens** Jupyter notebook lectures, found 
at `autolens_workspace/*/howtolens`. 

I recommend that you check them out if you are interested in more details!
"""
