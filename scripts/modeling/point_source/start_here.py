"""
Modeling: Start Here
====================

This script is the starting point for lens modeling of point-source lens datasets, for example the multiple image
positions of a lensed quasar.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing lens
modeling in different ways and customizing the analysis.

__Not Using Light Profiles__

Users familiar with PyAutoLens who are familiar with analysing imaging or interferometer data will be used to
performing lens modeling using light profiles, which have parameter that describe the shape and size of the
galaxy's luminous emission.

For point sources, for example a lensed quasar, it is invalid to model the source using light profiles, because they
implicitly assume an extended surface brightness distribution. Point source modeling instead assumes the source
has a (y,x) `centre` (y,x), but does not have other parameters like elliptical components or an effective radius.

This changes how the ray-tracing calculations that go into point source modeling are performed. They are briefly
touched on in this example, but for a more detailed explanation checkout the
`autolens_workspace/*/overview/overview_8_point_sources.py` example.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source `Point`.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` and `interferometer` examples.
For a quadruply imaged point source (8 data points) there is insufficient information to fully constain a model with
an `Isothermal` and `ExternalShear` (9 parameters).
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

Load the strong lens point-source dataset `simple`, which is the dataset we will use to perform point source 
lens modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "point_source", dataset_name)

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
We can also plot the positions and fluxes of the `PointDataset`.
"""
dataset_plotter = aplt.PointDatasetPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We next load an image of the dataset. 

Although we are performing point-source modeling and do not use this data in the actual modeling, it is useful to 
load it for visualization, for example to see where the multiple images of the point source are located relative to the 
lens galaxy.

The image will also be passed to the analysis further down, meaning that visualization of the point-source model
overlaid over the image will be output making interpretation of the results straight forward.

Loading and inputting the image of the dataset in this way is entirely optional, and if you are only interested in
performing point-source modeling you do not need to do this.
"""
data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.05
)

"""
We can also plot the dataset's multiple image positions over the observed image, to ensure they overlap the
lensed source's multiple images.
"""
visuals = aplt.Visuals2D(positions=dataset.positions)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

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
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's light is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

__Name Pairing__

Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` it will raise an error.

In this example, where there is just one source, name pairing appears unnecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDataset`.

__Coordinates__

The model fitting default settings assume that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/modeling/imaging/customize/priors.py`).
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# Source:

point_0 = af.Model(al.ps.Point)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

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

The lens model is fitted to the data using a non-linear search. 

All examples in the autolens workspace use the nested sampling algorithm 
Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/), which extensive testing has revealed gives the most 
accurate and efficient modeling results.

We make the following changes to the Nautilus settings:

 - Increase the number of live points, `n_live`, from the default value of 50 to 100. 

These are the two main Nautilus parameters that trade-off slower run time for a more reliable and accurate fit.
Increasing both of these parameter produces a more reliable fit at the expense of longer run-times.

__Customization__

The folders `autolens_workspace/*/point_source/modeling/searches` gives an overview of alternative non-linear searches,
other than Nautilus, that can be used to fit lens models. They also provide details on how to customize the
model-fit, for example the priors.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/point_source/modeling/simple/mass[sie]_source[point]/unique_identifier`.

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
"""
search = af.Nautilus(
    path_prefix=path.join("point_source", "modeling"),
    name="start_here",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
)

"""
__Chi Squared__

For point-source modeling, there are many different ways to define the likelihood function, broadly referred to a
an `image-plane chi-squared` or `source-plane chi-squared`. This determines whether the multiple images of the point
source are used to compute the likelihood in the source-plane or image-plane.

We will use an "image-plane chi-squared", which uses the `PointSolver` to determine the multiple images of the point
source in the image-plane for the given mass model and compares the positions of these model images to the observed
images to compute the chi-squared and likelihood.

There are still many different ways the image-plane chi-squared can be computed, for example do we allow for 
repeat image-pairs (i.e. the same multiple image being observed multiple times)? Do we pair all possible combinations
of multiple images to observed images? This example uses the simplest approach, which is to pair each multiple image
with the observed image that is closest to it, allowing for repeat image pairs. 

For a "source-plane chi-squared", the likelihood is computed in the source-plane. The analysis basically just ray-traces
the multiple images back to the source-plane and defines a chi-squared metric. For example, the default implementation 
sums the Euclidean distance between the image positions and the point source centre in the source-plane.

The source-plane chi-squared is significantly faster to compute than the image-plane chi-squared, as it requires 
only ray-tracing the ~4 observed image positions and does not require the iterative triangle ray-tracing approach
of the image-plane chi-squared. However, the source-plane chi-squared is less robust than the image-plane chi-squared,
and can lead to biased lens model results. If you are using the source-plane chi-squared, you should be aware of this
and interpret the results with caution.

Checkout the guide `autolens_workspace/*/guides/point_source.py` for more details and a full illustration of the
different ways the chi-squared can be computed.

__Analysis__

The `AnalysisPoint` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `PointDataset`.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # Image-plane chi-squared with repeat image pairs.
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

The log likelihood evaluation time can be estimated before a fit using the `profile_log_likelihood_function` method,
which returns two dictionaries containing the run-times and information about the fit.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.instance_from_prior_medians()
)

"""
The overall log likelihood evaluation time is given by the `fit_time` key.

For this example, it is ~1.0 second, which is a modest run-time. This is because the iterative ray-tracing approach
using triangles to compute the image-plane chi-squared is computationally quite slow. The source-plane chi-squared
is significantly faster to compute (below 0.1 seconds), but as discussed above is less robust.


"""
print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")

"""
To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform. 

Estimating this is tricky, as it depends on the lens model complexity (e.g. number of parameters)
and the properties of the dataset and model being fitted.

Testing has shown that point source datasets typically converge in fewer iterations than imaging or interferometer
datasets. For this example, we conservatively estimate that the non-linear search will perform ~1000 iterations per 
free parameter in the model. This is an upper limit, with models typically converging in far fewer iterations.

If you perform the fit over multiple CPUs, you can divide the run time by the number of cores to get an estimate of
the time it will take to fit the model. Parallelization with Nautilus scales well, it speeds up the model-fit by the 
`number_of_cores` for N < 8 CPUs and roughly `0.5*number_of_cores` for N > 8 CPUs. This scaling continues 
for N> 50 CPUs, meaning that with super computing facilities you can always achieve fast run times!
"""
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 1000)
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
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

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
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
