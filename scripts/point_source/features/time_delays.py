"""
Modeling: Time Delays
=====================

A measurable quantity of a point source is its time delay—the time it takes for light to travel from the source to the
observer for each multiple image of the point source (e.g., the quasar images). This is often expressed as the relative
time delay between each image and the image with the shortest time delay, which is often referred to as
the "reference image."

Time delays are commonly used in strong lensing analyses, for example to measure the Hubble constant, since
they are less affected by microlensing and can provide robust cosmological constraints.

This script describes how to perform point source lens modeling using the time delays of the point source dataset
as additional information on top of the positions of the point source, in case you are studying the Hubble constant
or another measureable quantity that uses time delays.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source `Point`.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` and `interferometer` examples.
For a quadruply imaged point source (8 data points) there is insufficient information to fully constain a model with
an `Isothermal` and `ExternalShear` (9 parameters).

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
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
dataset_plotter = aplt.PointDatasetPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We next load an image of the dataset and plot the point source data over it, because as described in 
the `modeling/start_here.ipynb` notebook, it is useful for visualizing the point source dataset.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.05)

visuals = aplt.Visuals2D(positions=dataset.positions)

array_plotter = aplt.Array2DPlotter(array=data, visuals_2d=visuals)
array_plotter.figure_2d()

"""
__Point Solver__

We set up the `PointSolver`, which is used to compute the multiple images of the point source in the image-plane.

There are no special settings or inputs for the fitting of time_delays, therefore the `PointSolver` is set up in the same way
as in the `modeling/start_here.ipynb` notebook.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
    xp=jnp,
)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 - The source galaxy's light is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

Name pairing is used as before to pair the `PointDataset` to the `Point` in the model, which is discussed below.

If you have fitted fluxes in the `fluxes` example, you will have seen that a `PointFlux` model component was used
which had the `flux` of the point source as an additional free parameter. For time delays, there is no special
model component or extra free parameters, because the time delays are a propety of the mass model.
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
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
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

Create the `AnalysisPoint` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairRepeat,  # Image-plane chi-squared with repeat image pairs.
)

"""
__Run Times__

For the positions-only fit, the run time of the log likelihood function was ~0.01 seconds, which is fast

Evaluating the time delays does not increase this much, with a value of around ~0.01 seconds still expected.

Overall modeling run times should therefore be around 20 minutes on CPU, under 5 minutes on GPU.

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

Time Delay lenses allow the Hubble constant to be constrained, because the difference between the geometric
time delay and the physical time delay is proportional to the Hubble constant.

We therefore create a Cosmology as a `Model` object in order to make the cosmological parameter Omega_m a free 
parameter.
"""
cosmology = af.Model(al.cosmo.FlatwCDMWrap)

"""
By default, all parameters of a cosmology model are initialized as fixed values based on the Planck18 cosmology.

In order to make the Hubble constant, we override the default value of the Hubble constant with uniform prior.
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
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
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
