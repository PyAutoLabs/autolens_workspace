"""
Modeling: Group Start Here Point
================================

This script models an example strong lens on the 'group' scale, where there is a single primary lens galaxy
and two smaller galaxies nearby, whose mass contributes significantly to the ray-tracing and is therefore included in
the strong lens model.

There are two methods for modeling group-scale strong lenses:

- `point`: The source is modeled as a point source, where the positions of its multiple images are fitted.
- `extended`: The source is modeled as an extended source, where the full emission is fitted, using imaging or interferometry data.

This example demonstrates the `point` method. For the `extended` source modeling, refer to
the `start_here_extended.ipynb` example.

Consider which method is appropriate for your dataset and the scientific goals of your analysis.

After completing this example, explore other relevant modeling packages for your
approach (e.g., `modeling/point_source` for point sources, `modeling/imaging` or `modeling/interferometer` for extended sources).

You may also want to explore the `features` package to extend the model for specific requirements (e.g., using a
pixelized source). Details on available features are provided at the end of this example.

__Scaling Relations__

This example models the mass of each galaxy individually, which means the number of dimensions of the model increases
as we model group scale lenses with more galaxies. This can lead to a model that is slow to fit and poorly constrained.

A common appraoch to overcome this is to put many of the galaxies a scaling relation, where the mass of the galaxies
are related to their light via a observationally motivated scaling relation. This means that as more galaxies are
included in the lens model, the dimensionality of the model does not increase.

Lens modeling using scaling relations is fully support and described in the `features/scaling_relation.ipynb` example,
you will likely want to read this example as soon as you have finished this one.

__Example__

This script fits a `PointDataset` dataset of a 'group-scale' strong lens where:

 - There is a main lens galaxy whose total mass distribution is an `Isothermal` and `ExternalShear`.
 - There are two extra lens galaxies whose total mass distributions are `IsothermalSph` models.
 - The source `Galaxy` is modeled as a point source `Point`.

The point-source dataset used in this example consists of the positions of the lensed source's multiple images and
their fluxes, however only the positions are used in the model-fit.

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
described in the script `autolens_workspace/*/data_preparation/imaging/start_here.ipynb`.
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

Load the strong lens dataset `group`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we perform point-source modeling and will not use this data in 
the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.

This is loaded via .fits files, which is a data format used by astronomers to store images.

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "group", dataset_name)

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.1
)

"""
We now load the point source dataset we will fit using point source modeling. 

We load this data as a `PointDataset`, which contains the positions and fluxes of every point source. 
"""
dataset = al.from_json(
    file_path=path.join(dataset_path, "point_dataset.json"),
)

"""
We can print this dictionary to see the dataset's `name`, `positions` and `fluxes` and noise-map values.
"""
print("Point Dataset Info:")
print(dataset.info)

"""
We can plot our positions dataset over the observed image.
"""
visuals = aplt.Visuals2D(positions=dataset.positions)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=dataset.positions)
grid_plotter.figure_2d()


"""
__Point Solver__

For point-source modeling we require a `PointSolver`, which determines the multiple-images of the mass model for a 
point source at location (y,x) in the source plane. 

It does this by ray tracing triangles from the image-plane to the source-plane and calculating if the 
source-plane (y,x) centre is inside the triangle. The method gradually ray-traces smaller and smaller triangles so 
that the multiple images can be determine with sub-pixel precision.

The `PointSolver` requires a starting grid of (y,x) coordinates in the image-plane which defines the first set
of triangles that are ray-traced to the source-plane. It also requires that a `pixel_scale_precision` is input, 
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

The `modeling/point_source` package provides full details of how the `PointSolver` works and the different
chi squared definitions available.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

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

for point source modeling, we do not model the light of the lens galaxies, as it is not necessary when only the 
positions of the multiple images are used to fit the model.

__Centres__

If the centres of the extra galaxies are treated as free parameters, one can run into the problem of having too many 
parameters and a model which is not fitted accurately.

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

 - The main lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - There are two extra lens galaxies with `IsothermalSph` total mass distributions, with centres fixed to the observed
   centres of light [2 parameters].
 
 - The source galaxy's light is a point `PointFlux` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
# Main Lens:

mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

# Extra Galaxies

extra_galaxies_centres = [(3.5, 2.5), (-4.4, -5.0)]

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Source:

point_0 = af.Model(al.ps.Point)
point_0.centre_0 = af.GaussianPrior(mean=0.0, sigma=3.0)
point_0.centre_1 = af.GaussianPrior(mean=0.0, sigma=3.0)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

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
__Name Pairing__

Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.

In this example, where there is just one source, name pairing appears unnecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDataset`!

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/modeling/point_source/customize/priors.py`).
"""
print(model)

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

The folders: 

 - `autolens_workspace/*/modeling/point_source/searches`.
 - `autolens_workspace/*/modeling/point_source/customize`
  
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

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
lens models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.

__Parallel Script__

Depending on the operating system (e.g. Linux, Mac, Windows), Python version, if you are running a Jupyter notebook 
and other factors, this script may not run a successful parallel fit (e.g. running the script 
with `number_of_cores` > 1 will produce an error). It is also common for Jupyter notebooks to not run in parallel 
correctly, requiring a Python script to be run, often from a command line terminal.

To fix these issues, the Python script needs to be adapted to use an `if __name__ == "__main__":` API, as this allows
the Python `multiprocessing` module to allocate threads and jobs correctly. An adaptation of this example script 
is provided at `autolens_workspace/scripts/modeling/point_source/customize/parallel.py`, which will hopefully run 
successfully in parallel on your computer!

Therefore if paralellization for this script doesn't work, check out the `parallel.py` example. You will need to update
all scripts you run to use the this format and API. 

__Iterations Per Update__

Every N iterations, the non-linear search outputs the current results to the folder `autolens_workspace/output`,
which includes producing visualization. 

Depending on how long it takes for the model to be fitted to the data (see discussion about run times below), 
this can take up a large fraction of the run-time of the non-linear search.

For this fit, the fit is very fast, thus we set a high value of `iterations_per_update=10000` to ensure these updates
so not slow down the overall speed of the model-fit.

**If the iteration per update is too low, the model-fit may be significantly slowed down by the time it takes to
output results and visualization frequently to hard-disk. If your fit is consistent displaying a log saying that it
is outputting results, try increasing this value to ensure the model-fit runs efficiently.**
"""
search = af.Nautilus(
    path_prefix=path.join("group", "modeling"),
    name="start_here_point",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__Analysis__

We next create an `AnalysisPoint` object, which can be given many inputs customizing how the lens model is 
fitted to the data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a lens model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autolens_workspace/*/point/log_likelihood_function`
"""
analysis = al.AnalysisPoint(dataset=dataset, solver=solver)

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

For this example, it is ~0.001 seconds, which is extremely fast for lens modeling. The source-plane chi-squared
is possibly the fastest way to fit a lens model to a dataset, and therefore whilst it has limitations it is a good
way to get a rough estimate of the lens model parameters quickly.
"""
print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
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
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
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

__Features__

The examples in the `autolens_workspace/*/modeling/features` package illustrate other lens modeling features. 

For point-source group scale modeling, we recommend you now checkout the following feature:

- ``scaling_relation.ipynb``: This feature allows you to model the light and mass of the extra galaxies using a scaling relation.

It is also recommended you read through the `point_source` package, to get a complete picture of how point-source 
modeling works.

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
