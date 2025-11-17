"""
Modeling: Group Start Here
==========================

This script models an example strong lens on the 'group' scale, which typically have a single "main" lens galaxy
and smaller extra galaxies nearby, whose light may blur with the source light and whose mass contributes significantly
to the ray-tracing, meaning both are therefore included in the strong lens model.

This example's lens has two extra galaxies whose light and mass are modeled, but this approach is expected to scale
well to group-scale systems with ~10 extra galaxies. Scaling above 10 is possible with **PyAutoLens**, but advanced
features described elsewhere in the workspace are likely required to ensure the model is accurate and efficient.

The source is modeled as an extended source, where the full emission is fitted. This example use imaging data
for illustration, but the same approach can be used for interferometer data.

Group scale lenses can be more complex than this example, for example with mutliple "main" lens galaxies, 20+ extra
galaxies, multiply sources. However, there is a point where the system should be regarded as a galaxy cluster-scale
lenses, and therefore techniques in the `cluster` section of the workspace should be used. The distinction between
group and cluster scale lenses is not well defined, so you may want to read the `cluster` section of the workspace too.

__Scaling Relations__

This example models the mass of each galaxy individually, which means the number of dimensions of the model increases
as we model group scale lenses with more galaxies. This can lead to a model that is slow to fit and poorly constrained.
There may also not be enough information in the data to constrain every galaxy's mass.

A common approach to overcome this is to put many of the extra galaxies a scaling relation, where the mass of the
galaxies are related to their light via a observationally motivated scaling relation. This means that as more
galaxies are included in the lens model, the dimensionality of the model does not increase. Furthermore, their
luminosities act as priors on their masses, which helps ensure the model is well constrained.

Lens modeling using scaling relations is fully support and described in the `features/scaling_relation.ipynb` example.
If your group has many extra galaxies (e.g. more than 5) you probably want to read this example once you are confident
with this one.

__Example__

This script fits an `Imaging` dataset of a 'group-scale' strong lens where

 - There is a main lens galaxy whose lens galaxy's light is an MGE.
 - There is a main lens galaxy whose total mass distribution is an `Isothermal` and `ExternalShear`.
 - There are two extra lens galaxies whose light models are `SersicSph` profiles and total mass distributions
   are `IsothermalSph` models.
 - The source galaxy's light is an MGE.

__Plotters__

To produce images of the data `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The `PLotter` API is described in the script `autolens_workspace/*/plot/start_here.py`.

__Simulation__

This script fits a simulated `Imaging` dataset of a strong lens, which is produced in the
script `autolens_workspace/*/imaging/simulator.py`

__Data Preparation__

The `Imaging` dataset fitted in this example confirms to a number of standard that make it suitable to be fitted in
**PyAutoLens**.

If you are intending to fit your own strong lens data, you will need to ensure it conforms to these standards, which are
described in the script `autolens_workspace/*/data_preparation/imaging/start_here.ipynb`.
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

"""
__Dataset__

Load the strong lens group dataset `simple`, which is the dataset we will use to perform lens modeling.

This is loaded via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "group", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

"""
Use an `ImagingPlotter` the plot the data, including: 

 - `data`: The image of the strong lens.
 - `noise_map`: The noise-map of the image, which quantifies the noise in every pixel as their RMS values.
 - `psf`: The point spread function of the image, which describes the blurring of the image by the telescope optics.
 - `signal_to_noise_map`: Quantifies the signal-to-noise in every pixel.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data. 

We create a 7.5 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
If we plot the masked data, the mask removes the exterior regions of the image where there is no emission from the 
lens and lensed source galaxies.

The mask used to fit the data can be customized, as described in 
the script `autolens_workspace/*/guides/modeling/customize`
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Main Galaxies and Extra Galaxies__

For a group-scale lens, we designate there to be two types of lens galaxies in the system:

 - `main_galaxies`: The main lens galaxies which likely make up the majority of light and mass in the lens system.
 These are modeled individually with a unique name for each, with their light and mass distributions modeled using 
 parametric models.
 
 - `extra_galaxies`: The extra galaxies which are nearby the lens system and contribute to the lensing of the source
  galaxy. These are modeled with a more restrictive model, for example with their centres fixed to the observed
  centre of light and their mass distributions modeled using a scaling relation. These are grouped into a single 
  `extra_galaxies` collection.
  
In this simple example group scale lens, there is one main lens galaxy and two extra galaxies. 

__Centres__

If the centres of the extra galaxies are treated as free parameters, there are too many 
parameters and the model may not be fitted accurately.

For group-scale lenses we therefore manually specify the centres of the extra galaxies, which are fixed to the observed
centres of light of the galaxies. `centre_1` and `centre_2` are the observed centres of the extra galaxies.

In a real analysis, one must determine the centres of the galaxies before modeling them, which can be done as follows:

 - Use the GUI tool in the `data_preparation/point_source/gui/extra_galaxies_centres.py` script to determine the centres
   of the extra galaxies. 

 - Use image processing software like Source Extractor (https://sextractor.readthedocs.io/en/latest/).

 - Fit every galaxy individually with a parametric light profile (e.g. an `Sersic`).

__Redshifts__

In this example all line of sight galaxies are at the same redshift as the lens galaxy, meaning multi-plane lensing
is not used.

If you have redshift information on the line of sight galaxies and some of their redshifts are different to the lens
galaxy, you can easily extend this example below to perform multi-plane lensing.

You would simply define a `redshift_list` and use this to set up the extra `Galaxy` redshifts.

__Model__

We compose a lens model where:

  - The main lens galaxy's light is a linear parametric `Sersic` bulge [6 parameters].

 - The main lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - There are two extra lens galaxies with linear `SersicSph` light and `IsothermalSph` total mass distributions, with 
   centres fixed to the observed centres of light [8 parameters].
 
 - The source galaxy's light is a point `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=27.

__Multi-Gaussian Expansion (MGE)__

A Multi-Gaussian Expansion (MGE) decomposes the lens and source light into ~50–100 Gaussians with varying ellipticities 
and sizes. An MGE captures irregular features far more effectively than Sérsic profiles, leading to more accurate lens m
odels.

Remarkably, modeling with MGEs is also significantly faster than using Sérsics: they remain efficient in JAX (on CPU 
or GPU), require fewer non-linear parameters despite their flexibility, and yield simpler parameter spaces that
sample in far fewer iterations. 

The MGE is extremely important for group-scale lenses. Every time we add an extra galaxy, the MGE does not add
any extra non-linear parameters, unlike light profiles like Sersics. This means we can model the light of many
extra galaxies, ensuring the lens light model is accurate, without making the model slow to fit or poorly constrained.

__Linear Light Profiles__

The MGE model below uses a **linear light profile** for the bulge and disk via the ``lp_linear`` API, instead of the 
standard ``lp`` light profiles used above.

A linear light profile solves for the *intensity* of each component via a linear inversion, rather than treating it as 
a free parameter. This reduces the dimensionality of the non-linear parameter space: a model with ~80 Gaussians
does not introduce ~80 additional free parameters.

Linear light profiles therefore improve speed and accuracy, and they are used by default in all modeling example.

__Model Composition__

The API below for composing a lens model uses the `Model` and `Collection` objects, which are imported from 
**PyAutoLens**'s parent project **PyAutoFit** 

The API is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the lens and source or using a different mass profile.

__Model Cookbook__

A full description of model composition is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Coordinates__

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/guides/modeling/customize`).
"""
# Main Lens:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

# Extra Galaxies

extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Light

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=10, centre_fixed=extra_galaxy_centre
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source:

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

For lensing calculations, the high magnification regions of a lensed source galaxy require especially high levels of 
over sampling to ensure the lensed images are evaluated accurately.

For a new user, the details of over-sampling are not important, therefore just be aware that calculations either:

 (i) use adaptive over sampling for the foregorund lens's light, which ensures high accuracy across. 
 (ii) use cored light profiles for the background source galaxy, where the core ensures low levels of over-sampling 
 produce numerically accurate but fast to compute results.

Over sampling at each extra galaxy centre is performed to ensure the lens calculations are accurate.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autolens_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
The imaging subplot updates the bottom two panels to reflect the update to over sampling, which now uses a higher
values in the centre.

Whilst you may not yet understand the details of over-sampling, you can at least track it visually in the plots
and later learnt more about it in the `over_sampling.ipynb` guide.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
The `info` attribute shows the model in a readable format.

This shows the group scale model, with separate entries for the main lens galaxy, the source galaxy and the 
extra galaxies.

The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).
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

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/group/simple/mass[sie]_source[point]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.

An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder. 

__Parallel Script__

Depending on the operating system (e.g. Linux, Mac, Windows), Python version, if you are running a Jupyter notebook 
and other factors, this script may not run a successful parallel fit (e.g. running the script 
with `number_of_cores` > 1 will produce an error). It is also common for Jupyter notebooks to not run in parallel 
correctly, requiring a Python script to be run, often from a command line terminal.

To fix these issues, the Python script needs to be adapted to use an `if __name__ == "__main__":` API, as this allows
the Python `multiprocessing` module to allocate threads and jobs correctly. An adaptation of this example script 
is provided at `autolens_workspace/scripts/guides/modeling/customize`, which will hopefully run 
successfully in parallel on your computer!

Therefore if paralellization for this script doesn't work, check out the `parallel.py` example. You will need to update
all scripts you run to use the this format and API. 

__Iterations Per Update__

Every N iterations, the non-linear search outputs the maximum likelihood model and its best fit image to the 
Notebook visualizer and to hard-disk.

This process takes around ~10 seconds, so we don't want it to happen too often so as to slow down the overall
fit, but we also want it to happen frequently enough that we can track the progress.

On GPU, a value of ~2500 will see this output happens every minute, a good balance. On CPU it'll be a little
longer, but still a good balance.
"""
search = af.Nautilus(
    path_prefix=Path("group", "modeling"),
    name="start_here",
    unique_tag=dataset_name,
    n_live=150,
)

"""
__Analysis__

We next create an `AnalysisImaging` object, which can be given many inputs customizing how the lens model is 
fitted to the data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a lens model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autolens_workspace/*/imaging/log_likelihood_function`

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
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

For this analysis, the log likelihood evaluation time is ~0.001 seconds, which is extremely fast for lens modeling. The source-plane chi-squared
is possibly the fastest way to fit a lens model to a dataset, and therefore whilst it has limitations it is a good
way to get a rough estimate of the lens model parameters quickly.

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

Estimating this is tricky, as it depends on the lens model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted.

For this example, we conservatively estimate that the non-linear search will perform ~10000 iterations per free 
parameter in the model. This is an upper limit, with models typically converging in far fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. Parallelization with Nautilus scales well, it speeds up the model-fit by the 
`number_of_cores` for N < 8 CPUs and roughly `0.5*number_of_cores` for N > 8 CPUs. This scaling continues 
for N> 50 CPUs, meaning that with super computing facilities you can always achieve fast run times!

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
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
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.

Checkout `autolens_workspace/*/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
It also contains information on the posterior as estimated by the non-linear search (in this example `Nautilus`). 

Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.

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

This is especially important for group scale modeling, in order to reduce the complexity of the model.

__Features__

The examples in the `autolens_workspace/*/modeling/features` package illustrate other lens modeling features. 

The examples in the `autolens_workspace/*/modeling/features` package illustrate other lens modeling features. 

We recommend you checkout the following features, because they make lens modeling in general more reliable and 
efficient (you will therefore benefit from using these features irrespective of the quality of your data and 
scientific topic of study).

We recommend you now checkout the following features:

- ``scaling_relation.ipynb``: This feature allows you to model the light and mass of the extra galaxies using a scaling relation.
- ``linear_light_profiles``: The model light profiles use linear algebra to solve for their intensity, reducing model complexity.
- ``multi_gaussian_expansion``: The lens (or source) light is modeled as ~25-100 Gaussian basis functions 
- ``pixelization``: The source is reconstructed using an adaptive RectangularMagnification or Voronoi mesh.
- ``no_lens_light``: The foreground lens's light is not present in the data and thus omitted from the model.

For group scale modeling, the multi Gaussian expansion is particularly important, as this can dramatically reduce the
dimensionality of the model and improve the accuracy of the fit for both the lens and source galaxies.

It is also recommended you read through the `imaging` package, to get a complete picture of how point-source 
modeling works.

__Data Preparation__

If you are looking to fit your own CCD imaging data of a strong lens, checkout  
the `autolens_workspace/*/data_preparation/imaging/start_here.ipynb` script for an overview of how data should be 
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
