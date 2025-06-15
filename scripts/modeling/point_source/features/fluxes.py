"""
Modeling: Fluxes
================

A measurable quantity of a point source is its flux—the total amount of light received from each multiple image
of the point source (e.g., the quasar images).

In practice, fluxes are often measured but not used directly when analyzing lensed point sources such as quasars or
supernovae. This is because fluxes can be significantly affected by microlensing, which many lens models do not
accurately capture. However, in this simulation, microlensing is not included, so the fluxes can be simulated and
fitted reliably.

Nevertheless, this script describes how to perform point source lens modeling using the fluxes of the point source
dataset as additional information on top of the positions of the point source, in case you are studying microlensing
or confident the fluxes are not affected by it.

__Model__

This script fits a `PointDataset` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `Isothermal`.
 - The source `Galaxy` is a point source with flux, a `PointFlux`.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` and `interferometer` examples.
For a quadruply imaged point source (8 data points) there is insufficient information to fully constain a model with
an `Isothermal` and `ExternalShear` (9 parameters).

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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

We load this data as a `PointDataset`, which contains the positions and fluxes of every point source. 
"""
dataset = al.from_json(
    file_path=dataset_path / "point_dataset_with_fluxes.json",
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

There are no special settings or inputs for the fitting of fluxes, therefore the `PointSolver` is set up in the same way
as in the `modeling/start_here.ipynb` notebook.
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
 - The source galaxy's light is a point `PointFlux` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.

Name pairing is used as before to pair the `PointDataset` to the `Point` in the model, which is discussed below.

To fit fluxes, our model point source also needs a flux parameter, which is done by using the `PointFlux`
component instead of the `Point` component. This has a free parameter `flux`, which is the flux of the point source
in the source-plane. 
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
The `info` attribute shows the model in a readable format, which now includes the `flux` parameter of the point source.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

In the `start_here.py` example 100 live points (`n_live=100`) were used to sample parameter space. We increase this
to 150, to account for the additional free parameters in the model that is the source flux.
"""
search = af.Nautilus(
    path_prefix=Path("point_source") / "modeling",
    name="fluxes",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
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

For the positions-only fit, the run time of the log likelihood function was ~0.4 seconds, which is a modest run-time.

Evaluating the flux does not increase this much, with a value of around ~0.5 seconds estimated, because evaluating the
fluxes is a simple multiplication of the magnification at each position by the flux of the point source.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
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
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that `flux` parameters of the source is inferred by the fit.
"""
print(result.info)

"""
Checkout `autolens_workspace/*/results` for a full description of analysing results in **PyAutoLens**.
"""
