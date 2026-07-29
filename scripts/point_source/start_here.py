"""
Start Here: Point Source
========================

Strong gravitational lenses often have point sources (e.g. quasars) that are being lensed, appearing as two or
four distinct point-like images. These lenses are particularly useful for measuring cosmological parameters
like the Hubble constant, and for studying the small-scale properties of dark matter.

This script shows you how to model such a lens system using **PyAutoLens** with as little setup
as possible. In about 15 minutes you’ll be able to point the code at your own data and
fit your first lens.

We model **real data**: the quadruply imaged quasar **RXJ1131-1231**, one of the most famous
lensed quasars in the sky. A background quasar at redshift z = 0.658 is lensed into four images by a
foreground elliptical galaxy at z = 0.295. Its four image positions were measured to
milli-arcsecond precision with HST imaging (Suyu et al. 2013), and its time delays were measured by
the COSMOGRAIL monitoring campaign (Tewes et al. 2013), making it a cornerstone of time-delay
cosmography — the technique that measures the Hubble constant from lensed quasars.

We focus on a *galaxy-scale* lens (a single lens galaxy). If you have multiple lens galaxies,
see the `group/start_here.ipynb` and `cluster/start_here.ipynb` examples.

Point source modeling uses the positions of the lensed source in the image-plane, and optionally may also
use their fluxes and time delays. Lensed quasars are also commonly observed with CCD imaging, which is
used to measure the point-source positions precisely; the extended arcs of the quasar's host galaxy in
such imaging can be modeled jointly with the point-source data, as shown in the
`multi/features/imaging_and_point_source` example.

__Contents__

- **JAX:** JAX acceleration for fast GPU/CPU model-fitting.
- **Google Colab Setup:** The introduction `start_here` examples are available on Google Colab, which allows you to run them.
- **Imports:** Import the required Python libraries.
- **Dataset:** Load and plot the strong lens dataset.
- **Point Solver:** For point-source modeling we require a `PointSolver`, which determines the multiple-images of the.
- **Model:** Compose the lens model fitted to the data.
- **Name Pairing:** The `PointDataset` above had a name, `point_0`.
- **Model Fit:** Perform the model-fit using the search and analysis.
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Result:** Overview of the results of the model-fit.
- **Model Your Own Lens:** If you have your own strong lens point source data, you are now ready to model it yourself by.
- **Fluxes and Time Delays:** If you have measured the fluxes and/or time delays of the lensed point sources, these can also be.
- **Simulator:** Let’s now switch gears and simulate our own strong lens point sources.
- **Sample:** Often we want to simulate *many* strong lenses — for example, to train a neural network or to.
- **Wrap Up:** Summary of the script and next steps.

__JAX__

PyAutoLens runs point-source model-fits on JAX by default. `AnalysisPoint`
auto-enables `use_jax=True` if you installed `autolens[jax]`; the search
driver wraps the likelihood in `jax.vmap(jax.jit(...))`.

For the broader JAX principles see `autolens_workspace/start_here.py`
`__JAX__`. For the most user-impactful piece — the `PointSolver(use_jax=True)`
+ `@jax.jit` pattern for fast forward solving — see the `__JAX Variant__`
at the end of `scripts/point_source/simulator.py`. Point-source solving
is the rare case where the `@jax.jit` wrap really pays off (the
triangle-refinement loop dominates simulation runtime); the variant
script shows how to do it cleanly.

__Google Colab Setup__

The introduction `start_here` examples are available on Google Colab, which allows you to run them in a web browser
without manual local PyAutoLens installation.

The code below sets up your environment if you are using Google Colab, including installing autolens and downloading
files required to run the notebook. If you are running this script not in Colab (e.g. locally on your own computer),
running the code will still check correctly that your environment is set up and ready to go.
"""

try:
    import google.colab
except ImportError:
    from autolens import setup_colab as _setup_colab
else:
    import importlib
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "autonerves", "--no-deps"]
    )
    _setup_colab = importlib.import_module("autonerves.setup_colab")

_setup_colab.for_autolens(
    raise_error_if_not_gpu=False  # Switch to False for CPU Google Colab
)

"""
__Imports__

Lets first import autolens, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.
"""
from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We begin by creating the point source dataset, which for now contains only:

1. The positions of the lensed images in the image-plane.
2. Their RMS noise-map values, corresponding to the uncertainty on their position measurements.

The positions below are the four quasar images of **RXJ1131-1231**, measured from HST imaging by
Suyu et al. (2013, ApJ 766, 70, Table 1). Following the PyAutoLens convention they are (y, x)
offsets in arcseconds, here centred on the lens galaxy G, and their RMS uncertainty is the 0.005"
astrometric precision quoted by that paper. The images are ordered A, B, C, D.

We print and plot the dataset to show these properties but also see that the dataset has a name,
this will be important later when we perform lens modeling.
"""
positions = al.Grid2DIrregular(
    [
        (-0.520, -2.037),  # Image A
        (0.662, -2.076),  # Image B
        (-1.632, -1.460),  # Image C
        (0.356, 1.074),  # Image D
    ]
)
noise_map = al.ArrayIrregular([0.005, 0.005, 0.005, 0.005])

dataset = al.PointDataset(
    name="point_0", positions=positions, positions_noise_map=noise_map
)

print("Point Dataset Info:")
print(dataset.info)

aplt.subplot_point_dataset(dataset=dataset)

"""
We save this dataset to the workspace `dataset` folder as .json files, so the rest of the script can
demonstrate the same loading API you will use for your own data. The .json files shipped with the
workspace are identical to the ones written here.
"""
dataset_name = "rxj1131"
dataset_path = Path("dataset") / "point_source" / dataset_name

if not (dataset_path / "point_dataset_positions_only.json").exists():
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / "point_dataset_positions_only.json",
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
When CCD imaging of the lens is available, it can be loaded alongside the point dataset for
visualization — seeing where the multiple images sit relative to the lens galaxy makes results much
easier to interpret, and an image passed to the analysis is overlaid in its output visuals. This is
entirely optional, and for RXJ1131 the joint modeling of its CCD imaging and point-source data is
covered in the `multi/features/imaging_and_point_source` example.

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
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
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

lens = af.Model(al.Galaxy, redshift=0.295, mass=mass)

# Source:

point_0 = af.Model(al.ps.Point)

source = af.Model(al.Galaxy, redshift=0.658, point_0=point_0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The redshifts above are the measured values for RXJ1131-1231: the elliptical lens galaxy is at
z = 0.295 and the quasar at z = 0.658. For position-only fitting the redshifts only set the order
of the planes, but once time delays are included (below) they matter physically, because the time
delay of each image depends on the angular diameter distances between us, the lens and the source.

We can print the model to show the parameters that the model is composed of.
"""
print(model.info)

"""
__Model Fit__

We now fit the data with the lens model using the non-linear fitting method and nested sampling algorithm Nautilus.

This requires an `AnalysisPoint` object, which defines the `log_likelihood_function` used by Nautilus to fit
the model to the point source data.

__Why Not MultiStartProdigy?__

The `start_here.py` examples for imaging and interferometer data fit with `af.MultiStartProdigy`, a much faster
multi-start gradient optimizer, and use `Nautilus` only in their `modeling.py` scripts where the full posterior
is needed.

That is not yet possible for point-source data: a gradient optimizer needs derivatives of the likelihood, and
the point-source likelihood is computed by solving the lens equation for the multiple image positions — an
operation PyAutoLens cannot yet differentiate through. Point-source fits therefore use `Nautilus` here too,
which has the compensation that you get the full posterior straight away.

__JAX__

`AnalysisPoint` defaults to `use_jax=True` when JAX is installed.
`AnalysisPoint._register_fit_point_pytrees()` runs on first `fit_from`
to register `FitPositionsSource`, `FitPositionsImagePair`, and `Tracer`
as JAX pytrees — you don't need to call `register_tracer_classes`
yourself for the modeling path (that's only required for the explicit
JIT-it-yourself pattern in `simulator.py`'s `__JAX Variant__`).

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may produce
an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.

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
    name="start_here",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder ("rxj1131").
    n_live=75,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see modeling examples for details.
    iterations_per_quick_update=250000,  # Every N iterations the max likelihood model is visualized and written to output folder.
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
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
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grid)
aplt.subplot_fit_point(fit=result.max_log_likelihood_fit)

"""
The result object contains pretty much everything you need to do science with your own strong lens, but details
of all the information it contains are beyond the scope of this introductory script. The `guides` and `result` 
packages of the workspace contains all the information you need to analyze your results yourself.

__Model Your Own Lens__

If you have your own strong lens point source data, you are now ready to model it yourself by adapting the code above
and simply writing your own `PointSourceDataset`, or loading one from .json if you have already created it.

A few things to note, with full details on data preparation provided in the main workspace documentation:

- PyAutoLens uses (y,x) conventions for all positions.
- Supply your own CCD image for the lensed quasar for visualization.
- Ensure the lens galaxy is roughly centered in the image.
- Double-check `pixel_scales` for your telescope/detector.
- Start with the default model — it works very well for pretty much all galaxy scale lenses!

__Time Delays__

Because the light-travel time along each image's path differs, variability of the quasar appears in
the four images at different times. These **time delays** were measured for RXJ1131 by the
COSMOGRAIL monitoring campaign (Tewes et al. 2013, A&A 556, A22), which observed the lens for nine
years: relative to image B, Δt_AB = 0.7 ± 1.4 days, Δt_CB = -0.4 ± 2.0 days and
Δt_DB = 91.4 ± 1.5 days.

Time delays can be included in the `PointDataset` and fitted by the lens model. Note that ordering
is shared across quantities, so the first time delay corresponds to the first position (image A)
and so on. PyAutoLens fits time delays *relative to the shortest delay*, so only the relative
values matter — we enter the delays relative to image B and give B itself the smallest measured
pairwise uncertainty (1.4 days), since it is the reference image.
"""
time_delays = al.ArrayIrregular(values=[0.7, 0.0, -0.4, 91.4])
time_delays_noise_map = al.ArrayIrregular(values=[1.4, 1.4, 2.0, 1.5])

dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=noise_map,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

if not (dataset_path / "point_dataset_with_time_delays.json").exists():
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / "point_dataset_with_time_delays.json",
    )

"""
__Fluxes__

The fluxes of the four images have also been measured, and the `PointDataset` accepts `fluxes` and
`fluxes_noise_map` inputs which are fitted using the model's magnification map (via the `PointFlux`
model component instead of `Point`).

You should think very carefully about whether including fluxes is sensible, even when you have the
data. Real lensed-quasar fluxes are affected by microlensing from stars in the lens galaxy, dust
extinction, and intrinsic source variability, all of which are difficult to model — RXJ1131 itself
shows strong microlensing, which is why we fit only its positions and time delays here.

__Model Fit__

Time delays do not need the model to be updated, as they are computed from the mass model and the
point source (y,x) position. The lens and source redshifts set the distances that convert the
model's Fermat potential differences into delays in days.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.295, mass=mass)

# Source:

point_0 = af.Model(al.ps.Point)

source = af.Model(al.Galaxy, redshift=0.658, point_0=point_0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=Path("point_source"),  # The path where results and output are stored.
    name="start_here_time_delay",  # The name of the fit and folder results are output to.
    unique_tag=dataset_name,  # A unique tag which also defines the folder ("rxj1131").
    n_live=75,  # The number of Nautilus "live" points, increase for more complex models.
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see VRAM section below.
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
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
The image can be saved to .fits for later use.
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
 - The RMS noise map of the positions, set to the centroid precision of PSF fitting on HST imaging
   (~5 mas) — *not* the imaging pixel scale, which is the detector's sampling rather than its
   centroiding precision.
 - The point source fluxes, by computing the magnification from the tracer and applying it to an
   input source flux.
 - The RMS noise map of the fluxes, set to 5% relative — for lensed quasars and supernovae,
   photometric flux uncertainties are dominated by microlensing systematics rather than photon noise.
 - The time delays, which come from the tracer's mass model.
 - The RMS noise of the time delays, set to 5% relative — matching COSMOGRAIL/TDCOSMO precision on
   well-sampled photometric monitoring of multiply-imaged quasars.

See `simulator.py` for a full discussion of these values.
"""
positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

time_delays = tracer.time_delays_from(grid=positions)

flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ArrayIrregular(values=fluxes)

position_noise = 0.005
flux_rel_noise = 0.05
time_delay_rel_noise = 0.05

positions_noise_map = al.ArrayIrregular([position_noise] * len(positions))

fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))

time_delays_noise_map = al.ArrayIrregular(
    values=np.abs(time_delays) * time_delay_rel_noise
)

"""
We can pass these to a `PointDataset` and output to hard disk as a .json file.
"""
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=positions_noise_map,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
    time_delays=time_delays,
    time_delays_noise_map=time_delays_noise_map,
)

aplt.subplot_point_dataset(dataset=dataset)

dataset_path = Path("dataset") / "point_source" / "simulated_lens"


al.output_to_json(
    obj=dataset,
    file_path=dataset_path / "point_dataset_positions_only.json",
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
print(al.model_util.SIMULATOR_RANDOM_LENS_SUMMARY)

"""
We now simulate a sample of strong lens, we just do 3 for efficiency here but you can increase this to any number.
"""
total_datasets = 3

for sample_index in range(total_datasets):

    lens_galaxy, source_galaxy = al.model_util.random_galaxies_for_simulation_from(
        include_lens_light=False, use_point_source=True
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    positions = solver.solve(
        tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
    )
    magnifications = al.LensCalc.from_tracer(
        tracer=tracer
    ).magnification_2d_via_hessian_from(grid=positions)
    time_delays = tracer.time_delays_from(grid=positions)

    flux = 1.0
    fluxes = [flux * np.abs(magnification) for magnification in magnifications]
    fluxes = al.ArrayIrregular(values=fluxes)

    positions_noise_map = al.ArrayIrregular([position_noise] * len(positions))
    fluxes_noise_map = al.ArrayIrregular(values=flux_rel_noise * np.asarray(fluxes))
    time_delays_noise_map = al.ArrayIrregular(
        values=np.abs(time_delays) * time_delay_rel_noise
    )

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
