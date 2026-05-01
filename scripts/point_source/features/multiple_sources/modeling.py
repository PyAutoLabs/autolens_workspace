"""
Modeling: Multiple Sources
==========================

This script fits a `PointDataset` of a 'galaxy-scale' strong lens with multiple lensed point sources at different
redshifts. The lens system is multi-plane: a foreground lens at z=0.5 deflects both background sources, while
source_0 at z=1.0 is itself a deflector for source_1 at z=2.0 (the "double Einstein cross" configuration). Each
source's multiple images are stored in their own `PointDataset`, and the two datasets are fitted jointly using
the multi/factor-graph API:

 - The foreground lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The first source `Galaxy` (at z=1.0) is itself a `Galaxy` with a mass profile and a `Point`.
 - The second source `Galaxy` (at z=2.0) is a `Point`-only galaxy.

Two `PointDataset`s are fitted simultaneously, one per lensed source. The fit uses one `AnalysisPoint` per
dataset, each wrapped in an `AnalysisFactor` that pairs it with the shared lens model. The factors are combined
into a `FactorGraphModel`, which sums the individual log-likelihoods to form the global log-likelihood the
non-linear search optimises. Multi-plane lensing is handled automatically inside `AnalysisPoint`, which uses each
`Point`'s plane redshift in the tracer when solving for image-plane positions.

This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for point-source modeling
(see `point_source/modeling.py`) and the multi/factor-graph API (see `multi/modeling.py` and `cluster/modeling.py`).
Common boilerplate is therefore not re-explained in detail here.

__Currently Blocked By PyAutoLens #480__

This script does not run end-to-end on the current PyAutoLens release. The `PointSolver` magnification filter
uses the tracer's last-plane magnification instead of the requested `plane_redshift`'s magnification, so every
likelihood evaluation finds 0 image positions for source_0 (whose plane z=1.0 is intermediate). See
https://github.com/PyAutoLabs/PyAutoLens/issues/480 — both this script and `simulator.py` are listed in
`config/build/no_run.yaml` until that bug is fixed. The script is left here in its intended form so the example
is correct as soon as #480 lands; no script changes will be needed once it does.

__Contents__

**Dataset:** Load the list of `PointDataset` objects, one per lensed source.
**Point Solver:** We set up the `PointSolver`, which determines the multiple images of each point source.
**Model:** Compose the multi-plane lens model fitted to the data.
**Name Pairing:** Each `PointDataset` name is paired with a `Point` model component of the same name.
**Search:** Configure the non-linear search used to fit the model.
**Analysis List:** Set up one `AnalysisPoint` per dataset.
**Analysis Factor:** Each analysis is wrapped in an `AnalysisFactor` paired with the shared lens model.
**Factor Graph:** All `AnalysisFactor` objects are combined into a `FactorGraphModel`.
**Model-Fit:** Pass the factor graph to the non-linear search.
**Result:** Iterate the per-analysis results returned by the factor-graph fit.

__Start Here Notebook__

If any code in this script is unclear, refer to the `point_source/modeling.ipynb` notebook for the single-source
case, and `multi/modeling.ipynb` for the factor-graph API used to combine multiple datasets.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens point-source dataset `multiple_sources`, which is the dataset we will fit. The simulator
writes one `PointDataset` per lensed source (`point_dataset_0.json` and `point_dataset_1.json`), one for each
source-plane redshift.
"""
dataset_name = "multiple_sources"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it is created by running the corresponding simulator script.
This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/features/multiple_sources/simulator.py"],
        check=True,
    )

"""
We load each point-source dataset as a `PointDataset` and place them into a list. Every entry in the list is
fitted by its own `AnalysisPoint` further down in the script.
"""
dataset_list = [
    al.from_json(file_path=dataset_path / f"point_dataset_{i}.json") for i in range(2)
]

for dataset in dataset_list:
    print("Point Dataset Info:")
    print(dataset.info)
    aplt.subplot_point_dataset(dataset=dataset)

"""
__Point Solver__

For point-source modeling we require a `PointSolver`, which determines the multiple images of the mass model for
a point source at location (y,x) in the source plane. It does this by ray tracing triangles from the image-plane
to the source-plane and refining the multiple images to sub-pixel precision.

The solver requires a starting grid of (y,x) image-plane coordinates and a `pixel_scale_precision` controlling
the precision of the converged multiple images. The grid below matches the simulator so the solver covers the
same region of sky used to generate the data.

Strong lens mass models have a "central image" which is nearly always significantly demagnified and not observed.
Setting `magnification_threshold=0.1` discards this image so it does not contaminate the fit.
"""
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
)

solver = al.PointSolver.for_grid(
    grid=grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

"""
__Model__

We compose a multi-plane lens model where:

 - The lens galaxy at z=0.5 has an `Isothermal` mass distribution with `ExternalShear` [7 parameters].

 - The first source galaxy at z=1.0 has its own `Isothermal` mass distribution and a `Point` source [7 parameters].
   The mass of this galaxy is what makes the system genuinely multi-plane: it lenses the further source behind it
   in addition to the foreground lens, doubling the number of images of source_1.

 - The second source galaxy at z=2.0 is a `Point` only [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.

__Name Pairing__

Every `PointDataset` carries a `name` (here `"point_0"` and `"point_1"`). This `name` pairs the dataset with the
`Point` in the model that has the same attribute name. Below, `source_0` has a `point_0` component which pairs
with `dataset_0`, and `source_1` has a `point_1` component which pairs with `dataset_1`.

If a model contains a `Point` whose name has no matching dataset, or vice versa, **PyAutoLens** raises an error.
The factor graph below ensures every dataset sees the full model, so the name pairs match across both analyses.

__Coordinates__

The model's prior centres assume the lens galaxy is near (0.0", 0.0"). If your dataset's lens is not at the
origin, recentre the data (`autolens_workspace/*/data_preparation`) or override the priors manually
(`autolens_workspace/*/guides/modeling/customize`).
"""
# Lens (z=0.5):

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)

# Source 0 (z=1.0) — itself a lens for source 1 behind it:

source_0 = af.Model(
    al.Galaxy,
    redshift=1.0,
    mass=al.mp.Isothermal,
    point_0=al.ps.Point,
)

# Source 1 (z=2.0):

source_1 = af.Model(al.Galaxy, redshift=2.0, point_1=al.ps.Point)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1)
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using a non-linear search. All examples in the autolens workspace use the nested
sampling algorithm Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/), which extensive testing has
revealed gives the most accurate and efficient modeling results.
"""
search = af.Nautilus(
    path_prefix=Path("point_source"),
    name="modeling",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Analysis List__

Set up one `AnalysisPoint` per dataset. We use an "image-plane chi-squared" via `FitPositionsImagePairRepeat`,
which is the most robust likelihood for point-source modeling. See `point_source/modeling.py` for an in-depth
discussion of the chi-squared options.

__JAX__

PyAutoLens uses JAX under the hood for fast GPU/CPU acceleration. If JAX is installed with GPU support, the fit
runs much faster (~10 minutes instead of an hour). On CPU-only systems JAX still provides a speed-up via
multithreading, with fits taking around 20-30 minutes.
"""
analysis_list = [
    al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairRepeat,
        use_jax=True,
    )
    for dataset in dataset_list
]

"""
__Analysis Factor__

Each analysis object is wrapped in an `AnalysisFactor`, which pairs it with the shared model and prepares it for
use in a factor graph. For this example every factor uses the same lens model, because all sources are lensed by
the same lens galaxy.

The term "Factor" comes from factor graphs, a type of probabilistic graphical model. In this context, each factor
represents the connection between one `PointDataset` and the shared model.
"""
analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]

"""
__Factor Graph__

All `AnalysisFactor` objects are combined into a `FactorGraphModel`, which represents a global model fit to
multiple datasets using a graphical model structure.

The key outcomes of this setup are:

 - The individual log likelihoods from each `Analysis` object are summed to form the total log likelihood
   evaluated during the model-fitting process.

 - Results from all datasets are output to a unified directory, with subdirectories for visualisations from each
   analysis object, as defined by their `visualize` methods.

This is a basic use of **PyAutoFit**'s graphical modeling capabilities, which support advanced hierarchical and
probabilistic modeling for large, multi-dataset analyses.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
To inspect the global model the factor graph fits, print `factor_graph.global_prior_model.info`.
"""
print(factor_graph.global_prior_model.info)

"""
__Model-Fit__

To fit multiple datasets, we pass the `FactorGraphModel` to a non-linear search.

Unlike single-dataset fitting, we now pass the `factor_graph.global_prior_model` as the model and the
`factor_graph` itself as the analysis object. This structure enables simultaneous fitting of multiple datasets
in a consistent and scalable way.

**Run Time Error:** On certain operating systems (e.g. Windows, Linux) and Python versions, the code below may
produce an error. If this occurs, see the `autolens_workspace/guides/modeling/bug_fix` example for a fix.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The result returned by a factor-graph fit is a list of `Result` objects, one per `AnalysisFactor`. Each entry
corresponds to the model-fit of one dataset. Because every factor sees the same global model, the
`max_log_likelihood_instance` is identical across results — the per-result objects differ only in their
analysis-specific data and visualisations.
"""
for result in result_list:
    print(result.max_log_likelihood_instance)

    aplt.subplot_tracer(
        tracer=result.max_log_likelihood_tracer,
        grid=grid,
    )

"""
The `Samples` object has the dimensions of the overall non-linear search and is identical in every result, so it
is sufficient to plot the corner from only the first result.
"""
aplt.corner_anesthetic(samples=result_list[0].samples)

"""
__Wrap Up__

This example introduces the API for fitting multiple lensed point sources at different redshifts with the
multi/factor-graph API. The same pattern can be extended to many more sources (see `cluster/modeling.py`) or to
combine point-source data with imaging or interferometer data via heterogeneous analyses in a single factor graph.
"""
