"""
Modeling: Start Here
====================

This script is the starting point for lens modeling of CCD imaging data (E.g. Hubble Space Telescope, Euclid) with
**PyAutoLens** and it provides an overview of the lens modeling API.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing lens
modeling in different ways and customizing the analysis.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `Sersic`.

This lens model is simple and computationally fast to fit, and therefore acts as a good starting point for new
users.

__Plotters__

To produce images of the data `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The `PLotter` API is described in the script `autolens_workspace/*/plot/start_here.py`.

__Simulation__

This script fits a simulated `Imaging` dataset of a strong lens, which is produced in the
script `autolens_workspace/*/imaging/simulators/start_here.py`

__Data Preparation__

The `Imaging` dataset fitted in this example confirms to a number of standard that make it suitable to be fitted in
**PyAutoLens**.

If you are intending to fit your own strong lens data, you will need to ensure it conforms to these standards, which are
described in the script `autolens_workspace/*/imaging/data_preparation/start_here.ipynb`.
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
__Dataset__

Load the strong lens dataset `simple` via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
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

The model-fit requires a `Mask2D` defining the regions of the image we fit the lens model to the data. 

Below, we create a 3.0 arcsecond circular mask and apply it to the `Imaging` object that the lens model fits.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
If we plot the masked data, the mask removes the exterior regions of the image where there is no emission from the 
lens and lensed source galaxies.

The mask used to fit the data can be customized, as described in 
the script `autolens_workspace/*/imaging/modeling/customize/mask.py`
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

In this example we compose a lens model where:

 - The lens galaxy's light is a parametric `Sersic` bulge and `Exponential` disk, the centres of 
 which are aligned [11 parameters].
 
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `Sersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

__Model Composition__

The API below for composing a lens model uses the `Model` and `Collection` objects, which are imported from 
**PyAutoLens**'s parent project **PyAutoFit** 

The API is is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the lens and source or using a different mass profile.

__Model Cookbook__

A full description of model composition, including lens model customization, is provided by the model cookbook: 

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Coordinates__

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/imaging/modeling/customize/priors.py`).
"""
# Lens:

bulge = af.Model(al.lp.Sersic)
disk = af.Model(al.lp.Exponential)
bulge.centre = disk.centre

mass = af.Model(al.mp.Isothermal)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk, mass=mass, shear=shear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.

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

The lens model is fitted to the data using a non-linear search. 

All examples in the autolens workspace use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/), which extensive testing has revealed gives the most accurate
and efficient  modeling results.

We make the following changes to the Dynesty settings:

 - Increase the number of live points, `nlive`, from the default value of 50 to 100. 
 - Increase the number of random walks per live point, `walks` from the default value of 5 to 10. 

These are the two main Dynesty parameters that trade-off slower run time for a more reliable and accurate fit.
Increasing both of these parameter produces a more reliable fit at the expense of longer run-times.

__Customization__

The folders `autolens_workspace/*/imaging/modeling/searches` gives an overview of alternative non-linear searches,
other than Dynesty, that can be used to fit lens models. They also provide details on how to customize the
model-fit, for example the priors.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/modeling/simple/light[bulge_disk]_mass[sie]_source[bulge]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Dynesty uses parallel processing to sample multiple 
lens models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=100,
    walks=10,
    number_of_cores=1,
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
the time it will take to fit the model. However, above ~6 cores the speed-up from parallelization is less efficient and
does not scale linearly with the number of cores.
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs the 
dynesty non-linear search in order to find which models fit the data with the highest likelihood.

__Output Folder__

Now this is running you should checkout the `autolens_workspace/output` folder.

This is where the results of the search are written to your hard-disk (in the `tutorial_1_non_linear_search` folder). 
When its completed, images, results and information about the fit appear in this folder, meaning that you don't need 
to keep running Python code to see the result.

__On The Fly Outputs__

Even when the search is running, information about the highest likelihood model inferred by the search so-far 
is output to this folder on-the-fly. 

If you navigate to the folder: 

 `output/howtolens/chapter_1/tutorials_1_non_linear_search/unique_identifier` 
 
Even before the search has finished, you will see:

 1) The `images` folder, where images of the highest likelihood lens model are output on-the-fly. This includes the
 `FitImaging` subplot we plotted in the previous chapter, which therefore gives a real sense of 'how good' the model
 fit is.
 
 2) The `samples` folder, which contains a `.csv` table of every sample of the non-linear search as well as other 
 information. 
 
 3) The `model.info` file, which lists the lens model, its parameters and their priors (discussed in the next tutorial).
 
 4) The `model.results` file, which lists the highest likelihood lens model and the most probable lens model with 
 errors (this outputs on-the-fly).
 
 5) The `search.summary` file, which provides a summary of the non-linear search settings and statistics on how well
 it is performing.
"""
result = search.fit(model=model, analysis=analysis)

"""
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
 - The corresponding maximum log likelihood `Plane` and `FitImaging` objects.
 - Information on the posterior as estimated by the `Dynesty` non-linear search. 
 
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
This script gives a concise overview of the PyAutoLens modeling API, fitting one the simplest lens models possible.
So, what next? 

__Features__

The following examples in the `modeling` package illustrate the advanced features of PyAutoLens's lens modeling:

- `autolens_workspace/*/imaging/modeling/features`: using PyAutoLens's advanced lens modeling 
  features, such as pixelized source reconstructions, multi Gaussian lens light models and fitting double Einstein rings.

- `autolens_workspace/*/imaging/modeling/searches`: using different non-linear searches to fit lens models, such as
  the MCMC algorithm `Emcee`, as well as how to customize the priors of the lens model.

- `autolens_workspace/*/imaging/modeling/customize`: customizing how the model is fitted to data via the analysis 
  object, for example using custom masks.

__HowToLens__

This `start_here.py` script, and the features examples above, do not explain many details of how lens modeling is 
performed, for example:

 - How does PyAutoLens perform ray-tracing and lensing calculations in order to fit a lens model?
 - How is a lens model fitted to data? What quantifies the goodness of fit (e.g. how is a log likelihood computed?).
 - How does dynesty find the highest likelihood lens models? What exactly is a "non-linear search"?

You do not need to be able to answer these questions in order to fit lens models with PyAutoLens and do science.
However, having a deeper understanding of how it all works is both interesting and will benefit you as a scientist

This deeper insight is offered by the **HowToLens** Jupyter notebook lectures, found 
at `autolens_workspace/*/howtolens`. 

I recommend that you check them out if you are interested in more details!

__Data Preparation__

You may now be looking to fit your own CD imaging data of a strong lens. 

In this case, you should checkout the `autolens_workspace/*/imaging/data_preparation/start_here.ipynb` script for an
overview of how data should be prepared before being modeled via **PyAutoLens**.
"""
