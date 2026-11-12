"""
Modeling: Time Delays
=====================

Use the relative arrival times of variability in multiply imaged quasars or supernovae, together with
image positions, to constrain the lens and infer cosmological distances. Time-delay cosmography uses
these measured delays and a lens model to constrain the Hubble constant, H0.

This example first fits positions and time delays at fixed cosmology. It then frees H0 while fixing the
lens centre and Einstein radius, illustrating how cosmology enters the model. This simplified second fit
is a teaching example: a cosmographic measurement must also account for uncertainty in the lens mass
profile, the lens environment and line of sight, and the measured delays.

A time delay is a difference in arrival time between images of the same source variation, not a measurement
of the absolute light-travel time. Different path lengths and gravitational potentials produce that difference.
Here the default fit subtracts the minimum delay from both the observed and predicted values; the analytic
reference-time alternative is explained below. These scripts fit already measured delays, rather than extracting
them from light curves. Microlensing and variability modeling can affect those measurements.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset:** Load and plot the strong lens dataset.
- **Point Solver:** Compute the multiple-image positions for a trial lens model.
- **Search:** Configure the non-linear search used to fit the model.
- **Analysis:** Create the Analysis object that defines how the model is fitted to the data.
- **Run Times:** Profiling the expected run time of the model-fit.
- **Result:** Overview of the results of the model-fit.
- **Cosmology:** Fit H0 in a simplified lens model and understand the assumptions.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a `PointSolved` point source whose centre is solved analytically.

This example uses an Isothermal lens without external shear to keep the demonstration simple. Whether a
more flexible model is constrained depends on the full dataset, priors and degeneracies; positions alone,
positions plus fluxes, and positions plus time delays carry different information.

When needed, external shear belongs in an `al.MassField` in the model's `fields` slot. See
`imaging/modeling.py` for that composition.

__Start Here Notebook__

If any code in this script is unclear, refer to the `point_source/start_here.ipynb` notebook.
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
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/point_source/simulator.py"
)

"""
We now load the point source dataset we will fit using point source modeling. 

We load this data as a `PointDataset`, which contains the positions and time_delays of every point source. 
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset_with_time_delays.json",
)

"""
We can print this dictionary to see the dataset's `name`, `positions` and `time_delays` and noise-map values.
"""
print("Point Dataset Info:")
print(dataset.info)

"""
We can also plot the positions of the `PointDataset`.
"""
aplt.subplot_point_dataset(dataset=dataset)

"""
We next load an image of the dataset, because as described in the `point_source/start_here.ipynb` notebook, it is
useful for visualizing the point source dataset.

It is also passed to the analysis below, where it is used only for visualization and not in the likelihood:
`image_with_points.png` shows the data with the observed positions (before the fit) and `fit_image_with_points.png`
adds the model positions and critical curves.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)


aplt.plot_array(array=data, title="")

"""
__Point Solver__

We set up the `PointSolver`, which is used to compute the multiple images of the point source in the image-plane.

There are no special settings or inputs for the fitting of time_delays, therefore the `PointSolver` is set up in the same way
as in the `point_source/start_here.ipynb` notebook.
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
 - The source galaxy is a `PointSolved` point source [0 parameters — its source-plane centre is solved
   analytically at every likelihood evaluation].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

Name pairing is used as before to pair the `PointDataset` to the `Point` in the model, which is discussed below.

If you have fitted fluxes in the `fluxes` example, you will have seen that a `PointFlux` model component was used
which had the `flux` of the point source as an additional free parameter. For the initial fixed-cosmology fit, adding time delays requires no new source parameter.
The predicted delays depend on the lens potential, geometry and cosmology; the later example frees H0.
"""
# Lens:

mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)

# Source (`PointSolved`: the source-plane centre is solved analytically, not sampled; time delays
# are computed from the tracer at the observed positions, so they need no centre parameter either):

point_0 = af.Model(al.ps.PointSolved)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free, 
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the 
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed 
parameter.
"""
af.ModelPlotter(model).figure()

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `point_source/start_here.py` for a
full description).
"""
search = af.Nautilus(
    path_prefix=Path("point_source") / "features",
    name="time_delays",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU lens model fits are batched and run simultaneously, see modeling examples
)

"""
__Analysis__

Create the `AnalysisPoint` object that evaluates the point-source likelihood for the search.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    image=data,  # Visualization only (`image_with_points.png`, `fit_image_with_points.png`).
    # Positions use the default `FitPositionsImagePairAllSolved` (all-to-all pairing, solved centre).
)

"""
__Analytic Reference Time (Solved)__

The default `FitTimeDelays` compares *relative* delays by subtracting the minimum delay from both
the data and the model (the "reference image" convention). An alternative is
`al.FitTimeDelaysSolved`, which instead solves a precision-weighted analytic reference time from
the data in closed form and analytically marginalizes over it — a smooth (and JAX-differentiable)
alternative to the min-subtraction. It is selected via `fit_time_delays_cls`, independently of the
positions fit (the model composition is unchanged):
"""
analysis_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    image=data,  # Visualization only (`image_with_points.png`, `fit_image_with_points.png`).
    fit_time_delays_cls=al.FitTimeDelaysSolved,  # Analytic reference time instead of min-subtraction.
)

"""
__Run Times__

Runtime depends on the point solver, model complexity, search settings and hardware. Measure the likelihood
cost and inspect search progress for your own setup; an additional observable can change both the cost of
one evaluation and the number of evaluations needed to explore the posterior.

__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).
"""
print(result.info)


"""
__Cosmology__

The delay between two images is the difference in their Fermat potentials multiplied by the time-delay
distance and divided by the speed of light. At fixed redshifts and other cosmological parameters, the
time-delay distance scales inversely with H0. Geometric and gravitational contributions both enter the
Fermat potential.

We create a `FlatLambdaCDM` model and free H0 below. Omega_m remains fixed. We also fix the lens centre
and Einstein radius, so this demonstration does not propagate uncertainty in those lens parameters into H0.
"""
cosmology = af.Model(al.cosmo.FlatLambdaCDM)

"""
By default, all parameters of a cosmology model are initialized as fixed values based on the Planck18 cosmology.

To make H0 a free parameter, we replace its fixed value with the uniform prior below.
"""
cosmology.H0 = af.UniformPrior(lower_limit=0.0, upper_limit=150.0)

# Overall Lens Model:

lens.mass.centre.centre_0 = 0.0
lens.mass.centre.centre_1 = 0.0
lens.mass.einstein_radius = 1.6

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), cosmology=cosmology
)


"""
The `info` attribute shows the model in a readable format

This confirms the model includes the Cosmology, which has the Hubble constant as a free parameter.
"""
print(model.info)

"""
The cosmology enters the model as a component of its own beside `galaxies`: `FlatLambdaCDM`, with `H0` free and 
`Om0`, `Tcmb0`, `Neff`, `m_nu` and `Ob0` fixed to their Planck18 values. The lens's `centre` and 
`einstein_radius` were fixed just above, so this fit samples the lens ellipticity and the Hubble constant and 
nothing else.
"""
af.ModelPlotter(model).figure()

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `point_source/start_here.py` for a
full description).
"""
search = af.Nautilus(
    path_prefix=Path("point_source") / "features",
    name="time_delays_hubble_constant2",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
)

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
Checkout `autolens_workspace/*/guides/results` for a full description of analysing results.
"""

"""
__Related Geometric Guide__

For a specialized approximation to point-source image geometry, see
[Witt-Wynne and the SIEP solver](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/misc/witt_wynne.ipynb).
That guide explains its assumptions and compares its predictions with the full lens solver. Its geometric
ellipse is a different use of ellipses from fitting a galaxy's light isophotes.
"""
