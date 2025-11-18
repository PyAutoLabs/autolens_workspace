"""
Start Here: Imaging
===================

Strong gravitational lenses often have point sources (e.g. quasars) that are being lensed, appearing as two or
four distinct point-like images. These lenses are particularly useful for measuring cosmological parameters
like the Hubble constant, and for studying the small-scale properties of dark matter.

This script shows you how to model such a lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own data and
fit your first lens.

We focus on a *galaxy-scale* lens (a single lens galaxy). If you have multiple lens galaxies,
see the `start_here_group.ipynb` and `start_here_cluster.ipynb` examples.

Point source modeling uses the positions of the lensed source in the image-plane, and optionally may also
use their fluxes and time delays. However, it is common for lensed quasar overall to be observed by CCD
imaging data, which is used to measure the positions of the point sources precisions and produes visuals
of the strong lens which aid its interpretation.

This script therefore also shows how to plot the CCD imaging of a point source lens, but does not use the
imaging data to constrain the lens model itself.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

We also show how to simulate strong lens point sources. This is useful for building machine learning
training datasets, or for investigating lensing effects in a controlled way.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

import subprocess
import sys

try:
    import google.colab

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autoconf", "--no-deps"]
    )
except ImportError:
    pass

from autoconf import setup_colab

setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
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

We begin by Creating the point source dataset, which for now contains only: 

1. The positions of the lensed images in the image-plane.
2. Their RMS noise-map values, corresponding to the uncertainty on their position measurements.
3. The PSF (Point Spread Function).

We print and plot the dataset to show these properties but also see that the dataset has a name,
this will be import later when we perform lens modeling.
"""
positions = al.Grid2DIrregular(
    [(-1.039, -1.038), (0.442, 1.608), (1.609, 0.442), (1.179, 1.179)]
)
noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])

dataset = al.PointDataset(
    name="point_0", positions=positions, positions_noise_map=noise_map
)

print("Point Dataset Info:")
print(dataset.info)

dataset_plotter = aplt.PointDatasetPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We can also load the dataset from the workspace `datasset` folder, which means the image we
load below is also available.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
We next load an image of the dataset. 

Although we are performing point-source modeling and do not use this data in the actual modeling, it is useful to 
load it for visualization, for example to see where the multiple images of the point source are located relative to the 
lens galaxy.

The image will also be passed to the analysis further down, meaning that visualization of the point-source model
overlaid over the image will be output making interpretation of the results straight forward.

Loading and inputting the image of the dataset in this way is entirely optional, and if you are only interested in
performing point-source modeling you do not need to do this.

We also plot the dataset's multiple image positions over the observed image, to ensure they overlap the
lensed source's multiple images.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

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

The solver has various settings which are set below to ensure for lens modeling the multiple images are computed
accurately, precisely and efficiently. These are described elsewhere in the workspace documentation.

The triangle ray-tracing method is fully compatible wit JAX and is significantly accelerated on the GPU.
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

To perform lens modeling we must define a lens model, describing the mass profile of the lens 
galaxy and point source model of the source galaxy.

A brilliant lens model to start with is one which uses aSingular Isothermal 
Ellipsoid (SIE) plus shear to model the lens mass and simply assumes the source is
a point source, with a `centre` (y,x) position that is a free parameter of the model.

__Name Pairing__

The `PointDataset` above had a name, `point_0`. This `name` pairs  the dataset to the `Point` in 
the model below, which is called `point_0`. 

If there is no point-source in the model that has the same name as a `PointDataset`, that data 
is not used in the model-fit. 

For galaxy scale lenses, where there is just one source galaxy, name pairing is unnecessary. 
However, cluster-scale strong lenses use the point source modeling API. These systems can have
over 100 source galaxies, and name pairing is necessary to ensure every point source in 
the lens model is fitted to its particular lensed images in the `PointDataset`.
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
We can print the model to show the parameters that the model is composed of.
"""
print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

This requires an `AnalysisPoint` object, which defines the `log_likelihood_function` used by Nautilus to fit
the model to the point source data.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU
support, your fits will run much faster (around 10 minutes instead of an hour). If only a CPU is available,
JAX will still provide a speed up via multithreading, with fits taking around 20-30 minutes.

If you don’t have a GPU locally, consider Google Colab which provides free GPUs, so your modeling runs are much faster.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce 
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
search = af.Nautilus(
    path_prefix=Path("point_source"),  # The path where results and output are stored.
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder.
    n_live=75,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_quick_update=250000,  # Every N iterations the max likelihood model is visualized and written to output folder.
)

# Hacky way to use JAX PointSolver, fix soon

solver_jax = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
    xp=jnp,
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver_jax,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

"""
The code below begins the model-fit. This will take around 10 minutes with a GPU, or 20-30 minutes with a CPU.

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
__Result__

Now this is running you should checkout the `autolens_workspace/output` folder, where many results of the fit
are written in a human readable format (e.g. .json files) and .fits and .png images of the fit are stored.

When the fit is complex, we can print the results by printing `result.info`.
"""
print(result.info)

"""
The result also contains the maximum likelihood lens model which can be used to plot the best-fit lensing information
and fit to the data.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitPointDatasetPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result` 
packages of the workspace contains all the information you need to analyze your results yourself.

__Model Your Own Lens__

If you have your own strong lens point source data, you are now ready to model it yourself by adapting the code above
and simply writing your own `PointSourceDataset`, or loading one from .json if you have already created it.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- PyAutoLens uses (y,x) conventions, so the positions below are y = 1.0", y = 2.0", x = 0.0" and x = 0.0".
- Supply your own CCD image for the lensed quasar for visualization.
- Ensure the lens galaxy is roughly centered in the image.
- Double-check `pixel_scales` for your telescope/detector.
- Start with the default model — it works very well for pretty much all galaxy scale lenses!
"""
positions = al.Grid2DIrregular(
    [(-1.039, -1.038), (0.442, 1.608), (1.609, 0.442), (1.179, 1.179)]
)
noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])

dataset = al.PointDataset(
    name="point_0", positions=positions, positions_noise_map=noise_map
)

"""
__Fluxes and Time Delays__

If you have measured the fluxes and/or time delays of the lensed point sources, these can also be included in
the `PointDataset` above and fitted by the lens model.

We first add fluxes, time delays and their RMS noise-map values to the dataset. Note that ordering is used across
quantities, so the first flux and time delay corresponds to the first position (1.0, 0.0) and so on.
"""
positions = al.Grid2DIrregular(
    [(-1.039, -1.038), (0.442, 1.608), (1.609, 0.442), (1.179, 1.179)]
)
fluxes = al.ArrayIrregular(values=[6.82, 55.16, 53.63, 100.62])
time_delays = al.ArrayIrregular(values=[-136.99, -176.85, -177.02, -176.74])

positions_noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])
fluxes_noise_map = al.ArrayIrregular(values=[1.0, 1.0, 1.0, 1.0])
time_delays_noise_map = al.ArrayIrregular(values=[-34.25, -44.21, -44.26, -44.19])

dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=positions_noise_map,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

"""
__Model__

When we add fluxes to the point dataset, we also need to updatre our model such that our point source
objects have their `flux` as a free parameter we fit for. The model API below does this, using the `PointFlux` 
component instead of the `Point` component. 

Time delays do not need the model to be updated, as they are computed from the mass model and the 
point source (y,x) position.

You should think very carefully if including fluxes in your modeling is a sensible idea, even if you have
the data available. For real lenses, they are often affected by microlensing, dust extinction, and
intrinsic variability of the source, all of which are difficult to model. 
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# Source:

point_0 = af.Model(al.ps.PointFlux)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Model Fit__

We now fit the model to the data using Nautilus, as before, but including
the fluxes and time delays in the `AnalysisPoint` object.
"""
search = af.Nautilus(
    path_prefix=Path("point_source"),  # The path where results and output are stored.
    name="start_here_flux_time_delay",  # The name of the fit and folder results are output to.
    unique_tag="example_point",  # A unique tag which also defines the folder.
    n_live=75,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # For fast GPU fitting lens model fits are batched and run simultaneously.
    iterations_per_full_update=20000,  # Every N iterations the results are written to hard-disk for inspection.
)
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
)

result = search.fit(model=model, analysis=analysis)

"""
__Simulator__

Let’s now switch gears and simulate our own strong lens point sources. This is a great way to:

- Practice lens modeling before using real data.
- Build large training sets (e.g. for machine learning).
- Test lensing theory in a controlled environment.

With each point source we'll also output CCD imaging of the source which is useful for visually
showing the lensing configuration.

To do this we need to define a 2D grid of (y,x) coordinates in the image-plane. This grid is
where we’ll evaluate the light from the lens and source galaxies.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
We now define a `Tracer` — this is the key object that combines all galaxies in the system
and computes how light rays are deflected.

- The lens galaxy has both light (a Sersic bulge) and mass (an isothermal profile + shear).
- The source galaxy has its own light (a SersicCore profile).

Together they define a strong lens system. The tracer will “ray-trace” our grid through
this mass distribution and generate a lensed image.
"""
source_centre = (0.0, 0.0)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=source_centre,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
    point_0=al.ps.PointFlux(centre=source_centre, flux=1.0),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Plotting the tracer’s image gives us a “perfect” view of the strong lens system, before
adding telescope effects.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
The image cna be saved to .fits for later use.
"""
image = tracer.image_2d_from(grid=grid)

dataset_type = "point_source"
dataset_name = "start_here_example"
dataset_path = Path("dataset") / dataset_type / dataset_name

al.output_to_fits(
    values=image.native,
    file_path=dataset_path / "image.fits",
    overwrite=True,
)

"""
__Simulator__

We now compute: 

 - The point source positions, reusing the `PointSolver` above.
 - The RMS noise map of the positions, which uses the `pixel_scale` of the CCD imaging data the quasar is observed on.
 - The point source fluxes, by computing the magnificaiton from the tracer and applying it to an input source flux.
 - The RMS noise map of the fluxes, which is the square root of the observed counts of each image.
 - The time delays, which comes from the tracer's mass model.
 - The RMS noise of the time delays, which is assumed to be 0.25 * their values but in real data uses the time delay estimate process.
"""
positions = solver.solve(
    tracer=tracer,
    source_plane_coordinate=source_galaxy.point_0.centre
)

magnifications = tracer.magnification_2d_via_hessian_from(grid=positions)

time_delays = tracer.time_delays_from(grid=positions)

flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

positions_noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])

fluxes_noise_map = al.ArrayIrregular(values=[np.sqrt(flux) for _ in range(len(fluxes))])

time_delays_noise_map = al.ArrayIrregular(values=time_delays * 0.25)

"""
We can pass these to a `PointDataset` and output to hard disk as a .json file.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=grid.pixel_scale,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

point_dataset_plotter = aplt.PointDatasetPlotter(
    dataset=dataset,
)
point_dataset_plotter.subplot_dataset()

dataset_path = Path("dataset") / "point_source" / "simulated_lens"


al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset.json",
)

"""
__Sample__

Often we want to simulate *many* strong lenses — for example, to train a neural network
or to explore population-level statistics.

This uses the model composition API to define the distribution of the light and mass profiles
of the lens and source galaxies we draw from. The model composition is a little too complex for
the first example, thus we use a helper function to create a simple lens and source model.

We then generate 3 lenses for speed, and plot their images so you can see the variety of lenses
we create.

Each lens is simulated as if it were observed with CD imaging, therefore with a PSF and noise-map.
"""
lens_model, source_model = al.model_util.simulator_start_here_model_from(
    include_lens_light=False, use_point_source=True
)

print(lens_model.info)
print(source_model.info)

"""
We now simulate a sample of strong lens, we just do 3 for efficiency here but you can increase this to any number.
"""
total_datasets = 3

for sample_index in range(total_datasets):

    lens_galaxy = lens_model.random_instance()
    source_galaxy = source_model.random_instance()

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
    )
    magnifications = tracer.magnification_2d_via_hessian_from(grid=positions)
    time_delays = tracer.time_delays_from(grid=positions)

    flux = 1.0
    fluxes = [flux * np.abs(magnification) for magnification in magnifications]
    fluxes = al.ArrayIrregular(values=fluxes)

    positions_noise_map = al.ArrayIrregular([0.05, 0.05, 0.05, 0.05])
    fluxes_noise_map = al.ArrayIrregular(
        values=[np.sqrt(flux) for _ in range(len(fluxes))]
    )
    time_delays_noise_map = al.ArrayIrregular(values=time_delays * 0.25)

    dataset = al.PointDataset(
        name=f"point_0",
        positions=positions,
        fluxes=fluxes,
        time_delays=time_delays,
        positions_noise_map=positions_noise_map,
        fluxes_noise_map=fluxes_noise_map,
        time_delays_noise_map=time_delays_noise_map,
    )

"""
__Wrap Up__

This script has shown how to model point source data of strong lenses, and simulate your own strong lenses.

Details of the **PyAutoLens** API and how lens modeling and simulations actually work were omitted for simplicity,
but everything you need to know is described throughout the main workspace documentation. You should check it out,
but maybe you want to try and model your own lens first!

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/*/point_source/modeling`: A full description of the lens modeling API and how to customize your model-fits.
- `autolens_workspace/*/point_source/simulator`: A full description of the lens simulation API and how to customize your simulations.
- `autolens_workspace/*/point_source/data_preparation`: How to load and prepare your own imaging data for lens modeling.
- `autolens_workspace/guides/results`: How to load and analyze the results of your lens model fits, including tools for large samples.
- `autolens_workspace/guides`: A complete description of the API and information on lensing calculations and units.
- `autolens_workspace/point_source/feature`: A description of advanced features for lens modeling, for example time delays, read this once you're confident with the basics!
"""
