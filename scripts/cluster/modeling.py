"""
Modeling: Cluster Start Here
============================

This script models an example strong lens on the 'cluster' scale, where there is a Brightest Cluster Galaxy (BCG),
large dark matter halo, 20 extra galaxies in the cluster whose collective mass contributes significantly to the
ray-tracing and 5 background source galaxies.

The primary method for modeling cluster scale strong lenses uses `point` source modeling, where each source is modeled
as a point source, where the positions of its multiple images are fitted (but not the extended emission observed at a
pixel level).

__Scaling Relations__

This example models the mass of the cluster galaxies by putting them on a scaling relation which links light (measured
luminosity) to mass. This means the number of dimensions of the model does not increase as we add more and more
galaxies to the cluster lens model. Given the largest clusters have 100+ galaxies, this avoids our model complexity
blowing up to 100 of free parameter and is therefore key.

__Example__

This script fits a `PointDataset` dataset of a 'cluster-scale' strong lens where:

 - There is a main Brightest Cluster Galaxy lens whose total mass distribution is an `Isothermal` and `ExternalShear`.
 - There is a large scale dark matter halo modeled as an `NFWSph`.
 - There are ten extra lens galaxies in the cluster whose total mass distributions are `DPIEPotential` models where
   their mass is linked to their light via a scaling relation.
 - There are 5 source galaxies modeled as point sources.

The point-source dataset used in this example consists of the positions of every lensed source's multiple images
(their fluxes are not used).

__Plotters__

To produce images of the data `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The `PLotter` API is described in the script `autolens_workspace/*/guides/plot`.

__Simulation__

This script fits a simulated cluster dataset of a strong lens, which is produced in the
script `autolens_workspace/*/cluster/simulator.py`

__Data Preparation__

The `Imaging` dataset fitted in this example confirms to a number of standard that make it suitable to be fitted in
**PyAutoLens**.

If you are intending to fit your own strong lens data, you will need to ensure it conforms to these standards, which are
described in the script `autolens_workspace/*/imaging/data_preparation/start_here.ipynb`.
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens dataset `cluster`, which is the dataset we will use to perform lens modeling.

We begin by loading a CCD image of the dataset. Although we perform point-source modeling and will not use this data in 
the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. 

The use of an image in this way is entirely optional, and if it were not included in the model-fit visualization would 
performed without the image.

This is loaded via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "cluster", dataset_name)

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.1)

"""
We now load the point source datasets we will fit using point source modeling. 

We load this data as a list of `PointDataset` object, which contains the positions of every point source. 
"""
dataset_list = []

for i in range(5):

    dataset = al.from_json(
        file_path=Path(dataset_path, f"point_dataset_{i}.json"),
    )

    dataset_list.append(dataset)

"""
We can print this dictionary to see the dataset's `name` and `positions` and noise-map values.
"""
for dataset in dataset_list:

    print("Point Dataset Info:")
    print(dataset.info)

"""
We can plot the positions of each dataset over the observed image.
"""
positions_list = []

for dataset in dataset_list:

    positions_list.append(dataset.positions)

visuals = aplt.Visuals2D(positions=positions_list)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=positions_list)
grid_plotter.figure_2d()

"""
__Centres__

The centre of every extra lens galaxy is used to compose the lens model, fixing their mass distributions
to their centres of light.

We load these centres below and plot them on the image to confirm they are located correctly and
cover all galaxies.
"""
extra_galaxies_centre_list = al.Grid2DIrregular(
    al.from_json(file_path=Path(dataset_path, "extra_galaxies_centre_list.json"))
)

visuals = aplt.Visuals2D(light_profile_centres=extra_galaxies_centre_list)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Luminosities__

We also need the luminosity of each galaxy, which in this example is the measured property we relate to mass via
the scaling relation.

We again uses the true values of the luminosities from the simulated dataset, but in a real analysis we would have
to determine these luminosities beforehand (see discussion above).

This could be other measured properties, like stellar mass or velocity dispersion.
"""
extra_galaxies_luminosity_list = al.from_json(
    file_path=Path(dataset_path, "extra_galaxies_luminosities.json")
)

"""
__Point Solver__

For point-source modeling we require a `PointSolver`, which determines the multiple-images of the mass model for a 
point source at location (y,x) in the source plane. 

It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
that the multiple images can be determine with sub-pixel precision.

The `PointSolver` requires an initial grid of (y, x) coordinates in the image plane (defined above), which defines the 
first set of triangles to ray trace spanning the whole cluster.It also requires that a `pixel_scale_precision` is input, 
which is the resolution up to which the multiple images are computed. The lower the `pixel_scale_precision`, the
longer the calculation, with the value of 0.001 below balancing efficiency with precision.

Strong lens mass models have a multiple image called the "central image". However, the image is nearly always 
significantly demagnified, meaning that it is not observed and cannot constrain the lens model. As this image is a
valid multiple image, the `PointSolver` will locate it irrespective of whether its so demagnified it is not observed.
To ensure this does not occur, we set a `magnification_threshold=0.1`, which discards this image because its
magnification will be well below this threshold.

If your dataset contains a central image that is observed you should reduce to include it in
the analysis.

__Chi Squared__

For point-source modeling, there are many different ways to define the likelihood function, broadly referred to a
an `image-plane chi-squared` or `source-plane chi-squared`. This determines whether the multiple images of the point
source are used to compute the likelihood in the source-plane or image-plane.

We will use an "image-plane chi-squared", which uses the `PointSolver` to determine the multiple images of the point
source in the image-plane for the given mass model and compares the positions of these model images to the observed
images to compute the chi-squared and likelihood.

The `point_source` package provides full details of how the `PointSolver` works and the different
chi squared definitions available.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=1.0,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1, xp=jnp
)

"""
__Main Galaxies and Extra Galaxies__

For a cluster-scale lens, we designate there to be the following lensing objects in the system:

 - `main_galaxies`: The main lens galaxies which are the brightest and highest mass galaxies in the lens system. In
 clusters they are often BCGs. These are modeled individually with a unique name for each, with their mass distributions 
 modeled using parametric models. The cluster scale dark matter halo is also tied to the BCG.
 
 - `extra_galaxies`: The extra galaxies which make up the cluster, whose masses individually don't contirbute too much
 lensing but they collectively contribute to the lensing of the source galaxies a lot. These are modeled with a
  more restrictive model, for example with their centres fixed to the observed centre of light and their mass 
  distributions modeled using a scaling relation. These are grouped into a single  `extra_galaxies` collection.
  
In this simple example cluster scale lens, there is one main lens galaxy and ten extra galaxies. 

for point source modeling, we do not model the light of the lens galaxies, as it is not necessary when only the 
positions of the multiple images are used to fit the model.

__Centres__

If the centres of the extra galaxies are treated as free parameters, there are too many 
parameters and the model may not be fitted accurately.

For cluster-scale lenses we therefore manually specify the centres of the extra galaxies (which we loaded above) which 
are fixed to the observed centres of light of the galaxies.

In a real analysis, one must determine the centres of the galaxies before modeling them, which can be done as follows:

 - Use the GUI tool in the `data_preparation/point_source/gui/extra_galaxies_centre_list.py` script to determine the centres
   of the extra galaxies. 

 - Use image processing software like Source Extractor (https://sextractor.readthedocs.io/en/latest/).

 - Fit every galaxy individually with a light profile (e.g. an `Sersic`).

__Redshifts__

In this example all galaxies are at the same redshift in the image-plane, meaning multi-plane lensing is not used.

If you have redshift information on the line of sight galaxies and some of their redshifts are different to the lens
galaxy, you can easily extend this example below to perform multi-plane lensing.

You would simply define a `redshift_list` and use this to set up the extra `Galaxy` redshifts.

__Model__

We compose a lens model where:

 - The main lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`  with a large
 - `NFWSph` dark matter halo [9 parameters].
 
 - There are ten extra lens galaxies with `DPIEPotentialSph` total mass distributions, with centres fixed to the 
   observed centres of light and masses linked to light via a scaling relation whose parameters are fitted 
   for [3 parameters].
 
 - There are five source galaxies whose light is a `Point` [10 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.
"""
# Main Lens:

lens_centre = (0.0, 0.0)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.GaussianPrior(mean=lens_centre[0], sigma=0.3)
mass.centre.centre_1 = af.GaussianPrior(mean=lens_centre[1], sigma=0.3)

shear = af.Model(al.mp.ExternalShear)
dark = af.Model(al.mp.NFWSph)

dark.centre.centre_0 = af.GaussianPrior(mean=lens_centre[0], sigma=0.3)
dark.centre.centre_1 = af.GaussianPrior(mean=lens_centre[1], sigma=0.3)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear, dark=dark)

# Extra Galaxies

ra_star = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
rs_star = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
b0_star = af.LogUniformPrior(lower_limit=1e5, upper_limit=1e7)
luminosity_star = 1e9

extra_galaxies_dict = {}

for i, extra_galaxy_centre, extra_galaxy_luminosity in enumerate(
    zip(extra_galaxies_centre_list, extra_galaxies_luminosity_list)
):

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = extra_galaxy_centre
    mass.ra = ra_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.rs = rs_star * (extra_galaxy_luminosity / luminosity_star) ** 0.5
    mass.b0 = b0_star * (extra_galaxy_luminosity / luminosity_star) ** 0.25

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_dict[f"extra_galaxy_{i}"] = extra_galaxy

# Source:

source_galaxies_dict = {}

for i, positions in enumerate(positions_list):

    positions_centre_y = np.mean(positions, axis=0)
    positions_centre_x = np.mean(positions, axis=1)

    point = af.Model(al.ps.Point)
    point.centre_0 = af.GaussianPrior(mean=positions_centre_y, sigma=3.0)
    point.centre_1 = af.GaussianPrior(mean=positions_centre_x, sigma=3.0)

    source = af.Model(al.Galaxy, redshift=1.0, **{f"point_{i}": point})

    source_galaxies_dict[f"source_{i}"] = source

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, **source_galaxies_dict),
    extra_galaxies=af.Collection(**extra_galaxies_dict),
)

"""
The `info` attribute shows the model in a readable format.

This shows the cluster scale model, with separate entries for the main lens galaxy, the source galaxies and the 
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
__Name Pairing__

Every point-source dataset in the `PointDataset` has a name, (e.g. `point_0`, `point_1`). This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.

In cluster lenses, point-source datasets may have many source galaxies in them, and name pairing is necessary to 
ensure every point source in the lens model is  fitted to its particular lensed images in the `PointDataset`!

The model fitting default settings assume that the BCG lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/guides/modeling/customize`).
"""
print(model)

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

__Iterations Per Update__

Every N iterations, the non-linear search outputs the maximum likelihood model and its best fit image to the 
Notebook visualizer and to hard-disk.

This process takes around ~10 seconds, so we don't want it to happen too often so as to slow down the overall
fit, but we also want it to happen frequently enough that we can track the progress.

On GPU, a value of ~2500 will see this output happens every minute, a good balance. On CPU it'll be a little
longer, but still a good balance.
"""
search = af.Nautilus(
    path_prefix=Path("cluster"),  # The path where results and output are stored.
    name="modeling",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model is visualized in the Jupter Notebook and output to hard-disk.
)

"""
__Analysis__

We next create  `AnalysisPoint` objects, which can be given many inputs customizing how the lens model is 
fitted to the data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

We create a list of analysis objects, one for each dataset, which means that the lens modeling will fit each
set of multiple images one-by-one and then sum their likelihoods. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a lens model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autolens_workspace/*/point/log_likelihood_function`

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis_list = [
    al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
    )
    for dataset in dataset_list
]

"""
__Analysis Factor__

Each analysis object is wrapped in an `AnalysisFactor`, which pairs each analysis it with the model.

For this simple cluster examples, the API below in a very simple way. However, the factor graph API below is used for
many advanced lens modeling tasks elsewhere in the workspace.
"""
analysis_factor_list = []

for analysis in analysis_list:

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

All `AnalysisFactor` objects are combined into a `FactorGraphModel`, which represents a global model fit to 
multiple datasets using a graphical model structure.

The key outcomes of this setup are:

 - The individual log likelihoods from each `Analysis` object are summed to form the total log likelihood 
   evaluated during the model-fitting process.

 - Results from all datasets are output to a unified directory, with subdirectories for visualizations 
   from each analysis object, as defined by their `visualize` methods.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
To inspect this new model, with extra parameters for each dataset created, we 
print `factor_graph.global_prior_model.info`.
"""
print(factor_graph.global_prior_model.info)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to 
   the dataset such that a log likelihood is returned.
 
 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.
   
For this analysis, the log likelihood evaluation time is < 1 seconds on CPU, < 0.02 seconds on GPU, which is 
fast for cluster scale lens modeling. 

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

For this model, this is typically around > iterations, meaning that this script takes < ? seconds, 
or ? minutes on CPU, or < ? seconds, or ? minute on GPU.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

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
print(result_list[0].info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` and `FitImaging` objects.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result_list[0].max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result_list[0].max_log_likelihood_tracer, grid=result_list[0].grids.lp
)
tracer_plotter.subplot_tracer()

"""
It also contains information on the posterior as estimated by the non-linear search (in this example `Nautilus`). 

Below, we make a corner plot of the "Probability Density Function" of every parameter in the model-fit.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
plotter = aplt.NestPlotter(samples=result_list[0].samples)
plotter.corner_anesthetic()

"""
This script gives a concise overview of the basic cluster modeling API, fitting one the simplest lens models possible.

Lets now consider what features you should read about to improve your cluster lens modeling, especially if you are aiming
to fit more complex models to your data.

__Data Preparation__

If you are looking to fit your own point source data of a strong lens, checkout  
the `autolens_workspace/*/data_preparation/point_source/README.rst` script for an overview of how data should be 
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
