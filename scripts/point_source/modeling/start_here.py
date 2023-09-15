"""
Modeling: Point-Source Mass Total
=================================

This script is the starting point for lens modeling of point-source lens datasets (E.g. the multiple image positions
of a lensed quasar) with **PyAutoLens** and it provides an overview of the lens modeling API.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing lens
modeling in different ways and customizing the analysis.

__Source Plane Chi Squared__

This example performs point-source modeling using a source-plane chi-squared. This means the likelihood of a model
is evaluated based on how close the multiple image positions it predicts trace to the centre of the point source
in the source-plane.

This is often regard as a less robust way to perform point-source modeling than an image-plane chi-squared, and it
means that other information about the multiple images of the point source (e.g. their fluxes) cannot be used. On
the plus side, it is much faster to perform modeling using a source-plane chi-squared.

Visualization of point-source modeling results are also limited, as the feature is still in development.

__Image Plane Chi Squared (In Development)__

An image-plane chi-squared is also available, however it is an in development feature with limitations. The main
limitation is that the solver for the image-plane positions of a point-source in the source-plane is not robust. It
often infers incorrect additional multiple image positions or fails to locate the correct ones.

This is because I made the foolish decision to try and locate the positions by ray-tracing squares surrounding the
image-plane positions to the source-plane and using a nearest neighbor based approach based on the Euclidean distance.
This contrasts standard implementations elsewhere in the literature, which use a more robust approach based on ray
tracing triangles to the source-plane and using whether the source-plane position lands within each triangle.

This will one day be fixed, but we have so far not found time to do so.

__Model__

This script fits a `PointDict` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source `PointSourceChi`.

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

Load the strong lens dataset `simple`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we are performing point-source modeling and will not use this
data in the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "point_source", dataset_name)

data = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "data.fits"), pixel_scales=0.05
)

"""
We now load the point source dataset we will fit using point source modeling. We load this data as a `PointDict`,
which is a Python dictionary containing the positions and fluxes of every point source. 

In this example there is just one point source, but point source model can be applied to datasets with any number 
of source's.
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

"""
We can print this dictionary to see the `name`, `positions` and `fluxes` of the dataset, as well as their noise-map values.
"""
print("Point Source Dict:")
print(point_dict)

"""
We can also just plot the positions and fluxes of the `PointDict`.
"""
point_dict_plotter = aplt.PointDictPlotter(point_dict=point_dict)
point_dict_plotter.subplot_positions()
point_dict_plotter.subplot_fluxes()

"""
We can also plot our positions dataset over the observed image.
"""
visuals = aplt.Visuals2D(positions=point_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's light is a point `PointSourceChi` [2 parameters].

The `PointSourceChi` model component indiciates that the chi-squared value of the point source is evaluated in the
source-plane. By changing this component to a `Point`, the chi-squared value would instead be evaluated in the
image-plane (this feature is in development and currently does not work well, as discussed at the top of this 
example script).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

__Name Pairing__

Every point-source dataset in the `PointDict` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.

In this example, where there is just one source, name pairing appears unecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDict`!
**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/*/data_preparation`). 
 - Manually override the lens model priors (`autolens_workspace/*/imaging/modeling/customize/priors.py`).
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# Source:

point_0 = af.Model(al.ps.PointSourceChi)

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
__Point Solver__

For point-source modeling we also need to define our `PointSolver`. This object determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

[For a source-plane chi-squared, we actually do not need to use a `PointSolver` at all, as its only purpose is to
find the multiple-images of a source in the image-plane.

Nevertheless, if you are feeling bold enough to try and use the current imaege-plage chi-squared feature, go ahead
and use the `PointSolver` below.]
"""
grid = al.Grid2D.uniform(shape_native=data.shape_native, pixel_scales=data.pixel_scales)

point_solver = al.PointSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Search__

The lens model is fitted to the data using a non-linear search. 

All examples in the autolens workspace use the nested sampling algorithm 
Nautilus (https://Nautilus.readthedocs.io/en/latest/), which extensive testing has revealed gives the most accurate
and efficient  modeling results.

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
    path_prefix=path.join("point_source"),
    name="mass[sie]_source[point]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=1,
)

"""
__Analysis__

The `AnalysisPoint` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `PointDataset`.
"""
analysis = al.AnalysisPoint(point_dict=point_dict, solver=None)

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
search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
