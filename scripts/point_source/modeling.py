"""
Modeling: Start Here
====================

This script is the starting point for lens modeling of point-source lens datasets, for example the multiple image
positions of a lensed quasar.

__Contents__

- **Not Using Light Profiles:** Users who are familiar with analysing imaging or interferometer data will be used to performing.
- **Model:** Compose the lens model fitted to the data.
- **Dataset:** Load and plot the strong lens dataset.
- **Point Solver:** For point-source modeling we require a `PointSolver`, which determines the multiple-images of the.
- **Model Composition:** Compose the lens model using the Model and Collection API.
- **Name Pairing:** Every point-source dataset in the `PointDataset` has a name, which in this example was `point_0`.
- **Coordinates:** Coordinate system assumptions for the model-fit.
- **Search:** Configure the non-linear search used to fit the model.
- **Unique Identifier:** In the path above, the `unique_identifier` appears as a collection of characters, where this.
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Chi Squared:** For point-source modeling, there are many different ways to define the likelihood function, broadly.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **VRAM Use:** When running AutoLens with JAX on a GPU, the analysis must fit within the GPU’s available VRAM.
- **Run Times:** Profiling the expected run time of the model-fit.
- **Output Folder Layout:** Description of the structure of the `output` folder where results are written.
- **Result:** Overview of the results of the model-fit.
- **Results:** Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
- **Modeling Customization:** The folders `autolens_workspace/*/guides/modeling/searches` gives an overview of alternative.

__Not Using Light Profiles__

Users who are familiar with analysing imaging or interferometer data will be used to
performing lens modeling using light profiles, which have parameter that describe the shape and size of the
galaxy's luminous emission.

For point sources, for example a lensed quasar, it is invalid to model the source using light profiles, because they
implicitly assume an extended surface brightness distribution. Point source modeling instead reduces the source to a
single (y,x) source-plane position, with no other parameters like elliptical components or an effective radius. In
this example even that position is not a free parameter — it is solved for analytically at every likelihood
evaluation (see `__Solved Source Centre__` below).

This changes how the ray-tracing calculations that go into point source modeling are performed. They are briefly
touched on in this example, but for a more detailed explanation checkout the
`autolens_workspace/*/point_source/start_here.py` example.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source `PointSolved`, whose source-plane centre is solved for analytically.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` and `interferometer` examples.
For a quadruply imaged point source (8 data points) there is insufficient information to fully constain a model with
an `Isothermal` and `ExternalShear` (9 parameters).
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens point-source dataset `simple`, which is the dataset we will use to perform point source 
lens modeling.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulator.py"],
        check=True,
    )

"""
We now load the point source dataset we will fit using point source modeling. 

We load this data as a `PointDataset`, which contains the positions of every point source. 
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
We can print this dictionary to see the dataset's `name`, `positions`and noise-map values.
"""
print("Point Dataset Info:")
print(dataset.info)

"""
We can also plot the positions of the `PointDataset`.
"""
aplt.subplot_point_dataset(dataset=dataset)

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
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

"""
We can also plot the dataset's multiple image positions over the observed image, to ensure they overlap the
lensed source's multiple images.
"""

aplt.plot_array(array=data, title="")

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
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].

 - The source galaxy is a `PointSolved` point source [0 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

__Model Composition__

The API below for composing a lens model uses the `Model` and `Collection` objects, which are imported from
**PyAutoLens**'s parent project **PyAutoFit**

The API is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the lens and source or using a different mass profile.

A full description of model composition is provided by the model cookbook:

https://pyautolens.readthedocs.io/en/latest/general/model_cookbook.html

__Solved Source Centre__

The `al.ps.PointSolved` component has no free parameters: for every trial mass model, the source-plane centre
that best fits the observed positions is solved for analytically (a precision-weighted mean of the back-traced
positions), rather than sampled as two free parameters. This is the recommended default because:

 - It removes 2 parameters per point source, which compounds quickly (a 5-source cluster drops 10 parameters).

 - The analytic centre makes the likelihood far better behaved for the non-linear search: in benchmark tests
   a gradient-based search converges on the solved likelihood but stalls below the true solution with free
   centres, and Nautilus converges faster.

 - Its posteriors on the mass parameters are not artificially narrowed by the analytic solve — in like-for-like
   tests the solved fit's `einstein_radius` error bars were slightly wider than the free-centre fit's, not tighter.

A free source centre (`al.ps.Point`, 2 extra parameters) is the right choice when the centre itself carries
information you want to keep or share:

 - Informative centre priors, e.g. from a light-profile fit of the quasar host galaxy.
 - A centre linked across multiple datasets (bands, epochs) that must share one consistent source position.
 - The source position is itself the science measurement.
 - Standardizable-candle fluxes (e.g. lensed supernovae): `PointSolved` forces the analytically-solved flux
   fit, whose flat-prior flux normalization discards a standard-candle prior on the intrinsic flux — compose
   `al.ps.PointFlux` with free parameters instead.

If you compose a centre-bearing `al.ps.Point` model, you must also pass a free-centre fit class to the
analysis (e.g. `fit_positions_cls=al.FitPositionsImagePairAll`) — the solved default raises an error rather
than silently ignoring the centre priors. See `guides/point_source_pairing.py` for the full option matrix.

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
 - Manually override the lens model priors (`autolens_workspace/*/guides/modeling/customize`).
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# Source:

point_0 = af.Model(al.ps.PointSolved)

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

Other data types fit their `start_here.py` with `af.MultiStartProdigy`, a much faster multi-start gradient
optimizer, and reserve `Nautilus` for the `modeling.py` example where the full posterior is needed. The
point-source likelihood is differentiable (the solved image positions carry an exact implicit gradient), and
benchmark tests show `af.MultiStartProdigy` converges on the solved-centre likelihood used here — but this
example uses `Nautilus` because the full posterior (parameter error bars) is usually the goal of a
point-source fit. Note that gradient searches only converge reliably with the solved source centre; with a
free centre they stall below the true solution even with many starts.

Nautilus has one main setting that trades-off accuracy and computational run-time, the number of `live_points`. 
A higher number of live points gives a more accurate result, but increases the run-time. A lower value give 
less reliable lens modeling (e.g. the fit may infer a local maxima), but is faster. 

The suitable value depends on the model complexity whereby models with more parameters require more live points. 
The default value of 200 is sufficient for the vast majority of common lens models. Lower values often given reliable
results though, and speed up the run-times. In this example, given the model is quite simple (N=21 parameters), we 
reduce the number of live points to 100 to speed up the run-time.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing
the `dataset_name` to the search's `unique_tag`.

__Live Visual Update__

By default the quick-update image is only written to disk. Set `live_visual_update=True` to also push it to a
live display surface:

- **Python script** — a matplotlib window opens automatically and refreshes with each quick update, so you can
  watch the fit converge without leaving your terminal.
- **Jupyter / Colab notebook** — the cell that ran `search.fit(...)` shows a single self-updating image that
  refreshes in place every `iterations_per_quick_update`.

The disk write (`fit.png`) always happens regardless of this flag. Set it to `False` (the default) if you just
want the on-disk output, or if you are running in a headless environment (e.g. an HPC cluster).
"""
search = af.Nautilus(
    path_prefix=Path("point_source"),  # The path where results and output are stored.
    name="modeling",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=100,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
    iterations_per_quick_update=10000,  # Every N iterations the max likelihood model, is visualized in the Jupter Notebook and output to hard-disk.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Chi Squared__

For point-source modeling, there are many different ways to define the likelihood function, broadly referred to a
an `image-plane chi-squared` or `source-plane chi-squared`. This determines whether the multiple images of the point
source are used to compute the likelihood in the source-plane or image-plane.

We will use an "image-plane chi-squared", which uses the `PointSolver` to determine the multiple images of the point
source in the image-plane for the given mass model and compares the positions of these model images to the observed
images to compute the chi-squared and likelihood.

There are still many different ways the image-plane chi-squared can be computed, differing in how the model's
multiple images are paired to the observed positions. The default, used here, is `FitPositionsImagePairAllSolved`:
an "all-to-all" pairing, where every model image is compared against every observed position via a smooth
probabilistic mixture, combined with the analytically-solved source centre described above.

All-to-all pairing is the default because of its robustness to imperfect position data. In truth-anchored
benchmark tests, when one true multiple image was missing from the dataset (e.g. lost under the lens galaxy's
light or below the detection limit), the alternative "repeat" pairing — which pairs each observed position with
its nearest model image — mis-ranked the true lens model by a log likelihood of order 10^5, because it has no
way to leave an unobserved model image unmatched. The all-to-all mixture absorbs a missing image gracefully and
recovered the truth cleanly on the same data. On clean data the two pairings give statistically equivalent
results at near-identical cost, so robustness decides the default.

The default's penalties for a mismatched image count fall straight out of its mixture likelihood, with no
knobs to tune. If the model produces *too few* images (an observed position with no model image nearby), that
position's penalty grows quadratically with its distance to the nearest model image — automatic and severe, as
under-prediction should be (you *saw* the image). If the model produces *too many* images, each extra one pays
only a mild logarithmic Occam factor — so the demagnified central image that almost every lens model predicts
(and observations almost never detect) is tolerated by construction, but note a bright spurious predicted image
is penalized just as gently, which is one reason to inspect the image-plane residuals of the max-likelihood
model before trusting a fit. See `guides/point_source_pairing.py` for the full discussion of both failure
modes and the stricter, tunable over-prediction policies.

For a "source-plane chi-squared", the likelihood is computed in the source-plane. The analysis just ray-traces
the observed image positions back to the source-plane and defines a chi-squared metric there. This is orders of
magnitude faster than the image-plane chi-squared (no iterative triangle solve), and with the modern tensor
weighting (`al.FitPositionsSourceSolved`, `weighting="jacobian"`) it is far more accurate than its traditional
reputation: the tensor maps each image's position noise through the full lensing Jacobian, whereas the
traditional scalar magnification weighting can catastrophically mis-rank models when one image is highly
magnified. Truth-anchored tests show the tensor-weighted fit ranks the true model first at galaxy and cluster
scale alike, while the scalar version preferred wrong models by thousands of log likelihood. The image-plane
chi-squared remains the most robust choice and is the demonstrated default; the tensor source-plane fit is the
recommended fast alternative for large samples or as an initialization stage.

Checkout the guide `autolens_workspace/*/point_source/fit` for more details and a full illustration of the
different ways the chi-squared can be computed, and `guides/point_source_pairing.py` for the full option
matrix with the benchmark evidence.

__Analysis__

We next create an `AnalysisPoint` object, which can be given many inputs customizing how the lens model is 
fitted to the data, which in this example includes the solver and the chi-squared method.

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 

It is not vital that you as a user understand the details of how the `log_likelihood_function` fits a lens model to 
data, but interested readers can find a step-by-step guide of the likelihood 
function at ``autolens_workspace/*/cluster/likelihood_function``

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    # The default fit is `al.FitPositionsImagePairAllSolved` (all-to-all image-plane chi-squared, solved
    # source centre). Pass `fit_positions_cls` to use another, e.g. `al.FitPositionsImagePairAll` for a
    # free source centre or `al.FitPositionsSourceSolved` for the fast tensor source-plane chi-squared.
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
__VRAM Use__

When running AutoLens with JAX on a GPU, the analysis must fit within the GPU’s
available VRAM. If insufficient VRAM is available, the analysis will fail with an
out-of-memory error, typically during JIT compilation or the first likelihood call.

Two factors dictate the VRAM usage of an analysis:

- The number of arrays and other data structures JAX must store in VRAM to fit the model
  to the data in the likelihood function. This is dictated by the model complexity and dataset size.
  For a MGE model its relatively low, but for other models (e.g. pixelized sources) it can be much higher.

- The `batch_size` sets how many likelihood evaluations are performed simultaneously.
  Increasing the batch size increases VRAM usage but can reduce overall run time,
  while decreasing it lowers VRAM usage at the cost of slower execution.

Before running an analysis, users should check that the estimated VRAM usage for the
chosen batch size is comfortably below their GPU’s total VRAM.

For a point solver with an image-plane chi squared and one set of positions with a single plane VRAM use is relatively
low (~0.1GB). For models with more planes and datasets with more multiple images it can be much higher (> 1GB going
beyond 10GB).

The method below prints the VRAM usage estimate for the analysis and model with the specified batch size,
it takes about 20-30 seconds to run so you may want to comment it out once you are familiar with your GPU's VRAM limits.
"""
analysis.print_vram_use(model=model, batch_size=search.batch_size)

"""
__Run Times__

Lens modeling can be a computationally expensive process. When fitting complex models to high resolution datasets 
run times can be of order hours, days, weeks or even months.

Run times are dictated by two factors:

 - The log likelihood evaluation time: the time it takes for a single `instance` of the lens model to be fitted to 
   the dataset such that a log likelihood is returned.
 
 - The number of iterations (e.g. log likelihood evaluations) performed by the non-linear search: more complex lens
   models require more iterations to converge to a solution.
   
For this analysis, the log likelihood evaluation time is < 0.001 seconds on GPU, ~0.01 seconds on CPU, which is 
extremely fast for lens modeling. 

To estimate the expected overall run time of the model-fit we multiply the log likelihood evaluation time by an 
estimate of the number of iterations the non-linear search will perform, which is around 10000 to 30000 for this model.

GPU run times are around 10 minutes, CPU run times are around 30 minutes.

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
__Output Folder Layout__

Now the fit is running you should checkout the `autolens_workspace/output` folder. This is where results are
written to hard-disk in human-readable formats — `.json`, `.csv`, `.png` and plain text.

As the fit progresses, results are written on the fly using the highest likelihood model found by the
non-linear search so far. This means you can inspect the model-fit as it runs, without waiting for the
non-linear search to terminate.

Each completed fit lives at a path like::

    output/point_source/<dataset_name>/modeling/<unique_hash>/
        files/                         <- JSON + CSV: loadable Python objects
            tracer.json                <- max log likelihood Tracer
            model.json                 <- fitted af.Collection model
            samples.csv                <- full Nautilus samples
            samples_summary.json       <- max log likelihood parameter values + errors
            samples_info.json          <- metadata about the samples
            search.json                <- non-linear search configuration
            settings.json              <- search settings
            cosmology.json             <- cosmology used for the fit
            covariance.csv             <- parameter covariance matrix
        image/                         <- PNG: point-source fit visualisations
            positions.png              <- observed vs model-predicted multiple-image positions
            fluxes.png                 <- observed vs model-predicted point-source fluxes
            tracer.png                 <- tracer image-plane and source-plane plots
        model.info                     <- human-readable model summary
        model.results                  <- human-readable fit summary
        search.summary                 <- search run summary
        search_internal/               <- internal files used to resume / visualise the search
        metadata                       <- run metadata

The `<unique_hash>` is a 32-character identifier derived from the model, search and dataset, so re-running the
same configuration resumes from the existing fit automatically.

__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""
print(result.max_log_likelihood_instance)

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grid)

"""
The result contains the full posterior information of our non-linear search, including all parameter samples, 
log likelihood values and tools to compute the errors on the lens model. 

There are built in visualization tools for plotting this.

The plot is labeled with short hand parameter names (e.g. `sersic_index` is mapped to the short hand 
parameter `n`). These mappings ate specified in the `config/notation.yaml` file and can be customized by users.

The superscripts of labels correspond to the name each component was given in the model (e.g. for the `Isothermal`
mass its name `mass` defined when making the `Model` above is used).
"""
aplt.corner_anesthetic(samples=result.samples)

"""
__Results__

Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.

__Modeling Customization__

The folders `autolens_workspace/*/guides/modeling/searches` gives an overview of alternative non-linear searches,
other than Nautilus, that can be used to fit lens models. 

They also provide details on how to customize the model-fit, for example the priors.
"""
