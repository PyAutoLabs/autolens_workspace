"""
Start Here: Cluster
===================

Cluster-scale strong lenses are made of:

 - One or more **Brightest Cluster Galaxies (BCGs)** and bright satellites, modelled individually with
   their own light and mass profiles.
 - **Tens to hundreds of lower-mass member galaxies**, whose collective mass perturbs the deflection
   field non-trivially but whose individual contributions are too weak to constrain on their own. These
   are modelled jointly on a luminosity-mass scaling relation, so the entire population shares just two
   free parameters regardless of how many members are in the catalogue.
 - **One or more cluster-scale dark matter halos** (``10^14 – 10^15`` M_sun), modelled with NFW-like
   profiles and not tied to any individual galaxy.
 - **Multiple background sources at different redshifts**, multiply imaged by the cluster — this makes
   cluster lensing a genuine multi-plane ray-tracing problem.

This script gets you fitting a real cluster-scale lens system in roughly 15 minutes. The example dataset
is a small multi-plane cluster (2 main galaxies + 10 scaling members + 1 host halo + 2 sources at
``z = 1.0`` and ``z = 2.0``) and is fully simulated, so you can run end-to-end without supplying your
own data.

For galaxy-scale lenses (a single dominant lens and a single source), start with
``start_here_imaging.ipynb`` instead.

__Contents__

- **JAX:** GPU/CPU acceleration; cluster fits take ~10 minutes on a GPU.
- **Beta Feature:** Cluster modeling is a beta feature — what works and what doesn't.
- **Google Colab Setup:** Bootstraps the environment when running on Colab.
- **Imports:** The libraries we'll use.
- **Dataset:** Load the CCD image and the per-source point datasets.
- **Model CSVs:** Load the named-galaxy mass + point CSVs written by the simulator.
- **Scaling Galaxies Table:** Load the 10 scaling-tier members' centres and luminosities from a CSV.
- **Point Solver:** Set up the image-plane multiple-image solver.
- **Cluster Components:** The four tiers of object that make up the model.
- **Model:** Compose the lens model fitted to the data.
- **Analysis + Factor Graph:** Combine the per-source analyses into one global fit.
- **Search:** Configure Nautilus, the non-linear search.
- **Model Fit:** Run the fit.
- **Result:** Inspect the maximum-likelihood model.
- **Wrap Up:** Where to go next.

__JAX__

**PyAutoLens** uses JAX under the hood for fast GPU/CPU acceleration. On a GPU the cluster fit below
runs in ~10 minutes; on CPU JAX still multi-threads the likelihood evaluation and the fit completes in
20–30 minutes. Google Colab provides free GPUs if you don't have one locally.

__Beta Feature__

Cluster modeling with **PyAutoLens** is in beta. Strengths:

 - JAX-accelerated image-plane chi-squared is over 50× faster than mainstream cluster modeling tools.
 - Multi-plane ray tracing of arbitrary complexity is supported natively.
 - Hand-editable CSV inputs (point datasets, scaling-galaxy catalogues) make iterating on a real cluster
   straightforward.

Known limitations:

 - Default ``aplt`` visualization is tuned for galaxy-scale lenses; cluster-specific plotters are in
   active development.
 - Workspace documentation for cluster modeling is less comprehensive than for galaxy-scale features.

__Google Colab Setup__

The ``start_here`` examples are runnable on Google Colab without local PyAutoLens installation. The
block below installs the dependencies and downloads the example dataset if you're on Colab; running it
locally is a no-op.
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
    raise_error_if_not_gpu=False  # Switch to True to require GPU on Colab.
)

"""
__Imports__
"""
from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We load the simulated cluster dataset. The dataset folder contains:

 - ``data.fits`` / ``noise_map.fits`` / ``psf.fits`` — CCD imaging of the cluster (used for visualization).
 - ``point_datasets.csv`` — one row per observed multiple image, grouped by source ``name``, with a
   ``redshift`` column per source.
 - ``scaling_galaxies.csv`` — one row per scaling-tier member with columns ``y, x, luminosity``.
 - ``main_lens_centres.json`` — centres of the 2 individually-modelled main galaxies.
 - ``host_halo_centre.json`` — centre of the host halo.

If the dataset is missing on disk, the corresponding simulator script runs automatically.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "cluster" / dataset_name

if (
    not (dataset_path / "data.fits").exists()
    or not (dataset_path / "scaling_galaxies.csv").exists()
    or not (dataset_path / "mass.csv").exists()
):
    subprocess.run(
        [sys.executable, "scripts/cluster/simulator.py"],
        check=True,
    )

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.1)

aplt.plot_array(array=data, title="")

"""
__Point Datasets__

The per-source point datasets are loaded from a single hand-editable CSV. ``al.list_from_csv`` returns a
``List[PointDataset]`` where each entry carries the source's ``positions``, ``positions_noise_map``, and
``redshift`` (different per source — this is a multi-plane system).

In a real analysis you would replace ``point_datasets.csv`` with the multiple-image positions measured
from your own imaging (e.g. via PSF-fitting). The CSV is spreadsheet-editable: positions, noises, and
redshifts can be tweaked without touching Python.
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

for dataset in dataset_list:
    print("Point Dataset Info:")
    print(dataset.info)
    print(f"Redshift: {dataset.redshift}")

for dataset in dataset_list:
    aplt.plot_grid(
        grid=al.Grid2DIrregular(np.atleast_2d(dataset.positions)),
        title=dataset.name,
    )

"""
__Model CSVs__

The simulator writes the truth model into three family-level CSVs — ``mass.csv``, ``light.csv``,
``point.csv`` — keyed by a ``galaxy`` column with ``profile_class`` dispatch. See
``scripts/cluster/csv_api.py`` for the schema walkthrough.

Point-source modeling only needs the mass and point families (light profiles don't affect lensing).
Observed galaxy-light centres are treated as ground truth — they remove a large block of degenerate
parameters that the multiple-image positions alone cannot constrain. In a real analysis these centres
come from light-profile fits to the imaging data or external source catalogues.
"""
mass_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "mass.csv", family="mass"
)
point_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "point.csv", family="point"
)

"""
__Scaling Galaxies Table__

The 10 scaling-tier members come from ``scaling_galaxies.csv`` — one row per member with columns
``y, x, luminosity``. ``al.galaxy_table_from_csv`` returns a typed ``GalaxyTable`` with ``.centres``
(a ``Grid2DIrregular``) and ``.luminosities`` (a list). Adding more members to a real cluster is a
CSV-level edit: append rows, save, re-run. The number of free parameters in the model does not change.

In a real analysis the luminosities come from a prior light-only fit (e.g. an MGE bulge fit, or a SLaM
``source_lp_0`` stage). See ``scripts/group/features/scaling_relation/modeling_for_luminosities.py``
for the standalone-fit pattern.
"""
scaling_galaxies_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_galaxies_centres = scaling_galaxies_table.centres
scaling_galaxies_luminosity_list = scaling_galaxies_table.luminosities

print(f"Scaling galaxies: {len(scaling_galaxies_luminosity_list)} members")

"""
__Point Solver__

Point-source modeling needs a ``PointSolver`` to find the image-plane multiple images of each source.
The solver ray-traces triangles from the image plane back to the source plane, iteratively refining
until the requested precision is reached. We use the same configuration as the more detailed
``cluster/modeling.py``: a 100x100 starting grid, 0.001" precision, and a magnification threshold of
0.1 to discard heavily-demagnified central images.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=1.0)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Cluster Components__

The model has four tiers, one per cluster component:

 - **Main lens galaxies (2):** individually-modelled ``dPIEMassSph`` profiles with centre fixed to the
   observed light centres and free ``ra``, ``rs``, ``b0``. **6 free parameters total.**

 - **Scaling-tier members (10):** ``dPIEMassSph`` profiles with centre fixed to the CSV centres,
   ``ra`` and ``rs`` fixed at the simulator truth values (0.1" and 10.0"), and ``b0`` derived from the
   *shared* relation ``b0 = scaling_factor * luminosity ** scaling_exponent`` plus the per-member
   luminosity. **2 free parameters total for the whole tier — independent of the number of members.**

 - **Host dark matter halo:** a standalone ``Galaxy`` carrying an ``NFWMCRLudlowSph`` halo with
   centre fixed and a free ``mass_at_200``. **1 free parameter.**

 - **Source galaxies (2):** ``Point`` models, redshift pinned to each source's per-dataset value, with
   ``GaussianPrior`` centre priors initialised from the mean of each source's observed positions.
   **4 free parameters total.**

**Total: N = 13 free parameters.** Adding more rows to ``scaling_galaxies.csv`` does not grow N — that's
the defining feature of cluster-scale modeling on a scaling relation.

__Redshifts__

The two sources sit at different redshifts (``z = 1.0`` and ``z = 2.0``); the ``Tracer`` automatically
ray-traces through both source planes when solving the further source. Lens galaxies (main + scaling)
and the host halo all sit at ``z = 0.5``. ``NFWMCRLudlowSph`` needs ``redshift_source`` to evaluate the
Ludlow et al. (2016) concentration-mass relation — we anchor it to the *furthest* source, matching the
simulator convention.

__Model__

The model is composed below in four blocks: main-tier loop, host halo, source-tier loop, scaling-tier
loop (defining shared ``scaling_factor`` / ``scaling_exponent`` once outside the loop). The four
blocks are then bundled into a single ``af.Collection`` model that the analysis will receive.
"""
redshift_lens = 0.5
source_redshifts = [dataset.redshift for dataset in dataset_list]

# Build af.Model[Galaxy] instances directly from the family CSVs. Concrete CSV
# values become fixed af.Model defaults; we then promote selected params to
# priors below. Keys: lens_0, lens_1, host_halo, source_0, source_1.

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

# Main Lens Galaxies: free dPIE ra / rs / b0; centre stays fixed at the CSV value.
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.ra = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)
    galaxy_models[name].mass.rs = af.UniformPrior(lower_limit=5.0, upper_limit=40.0)
    galaxy_models[name].mass.b0 = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)

# Host Halo: free mass_at_200; centre + redshift_object + redshift_source fixed.
galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

# Source Galaxies: free Point centres with GaussianPrior initialised from the
# mean of each source's observed multiple-image positions (NOT the truth from
# point.csv — in a real analysis the truth is unknown).
for i, dataset in enumerate(dataset_list):
    positions = np.atleast_2d(dataset.positions)
    point_attr = getattr(galaxy_models[f"source_{i}"], f"point_{i}")
    point_attr.centre_0 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 0])), sigma=3.0
    )
    point_attr.centre_1 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 1])), sigma=3.0
    )

# Scaling Tier (shared scaling_factor + scaling_exponent; per-member b0 derived).

scaling_factor = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
scaling_exponent = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

scaling_ra_fixed = 0.1
scaling_rs_fixed = 10.0

scaling_galaxies_list = []
for centre, luminosity in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosity_list
):
    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.ra = scaling_ra_fixed
    mass.rs = scaling_rs_fixed
    mass.b0 = scaling_factor * luminosity**scaling_exponent

    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

scaling_galaxies = af.Collection(scaling_galaxies_list)

# Overall Model

model = af.Collection(
    galaxies=af.Collection(**galaxy_models),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Analysis + Factor Graph__

We create one ``AnalysisPoint`` per dataset. Each analysis owns its dataset's log-likelihood; the
factor graph combines them all into a single global model fit. The total log likelihood is the sum of
the per-dataset log likelihoods.

The factor-graph API is what enables cluster-scale modeling with multiple sources at different
redshifts — every source's positions contribute to the same global model, and the multi-plane
ray-tracing happens inside each dataset's likelihood evaluation.
"""
analysis_list = [
    al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True)
    for dataset in dataset_list
]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
__Search__

We use Nautilus, a robust nested-sampling algorithm. ``n_live=100`` is a sensible default for a 13-D
model — increase it for more complex clusters. ``n_batch=50`` batches the GPU log-likelihood
evaluations for throughput.

Results are written to ``autolens_workspace/output/cluster/simple/start_here/<unique_hash>/``. The
``unique_hash`` is generated from the model, search settings, and dataset — re-running with the same
configuration resumes the existing fit.
"""
search = af.Nautilus(
    path_prefix=Path("cluster"),
    name="start_here",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=2500,
)

"""
__Model Fit__

The fit takes ~10 minutes on a GPU and 20–30 minutes on CPU. Watch ``autolens_workspace/output`` for
on-the-fly visualization of the best-fit model.

**Run Time Error:** On certain operating systems and Python versions, the code below may produce an
error. If this occurs, see ``autolens_workspace/guides/modeling/bug_fix``.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed — this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run — you may now continue the notebook.")

"""
__Result__

``search.fit`` on a factor graph returns one ``Result`` per dataset. They share the same global
maximum-likelihood model but each carries its own per-dataset visualization and ``FitPoint`` object.
"""
for result in result_list:
    print(result.max_log_likelihood_instance)

    aplt.subplot_tracer(
        tracer=result.max_log_likelihood_tracer,
        grid=grid,
    )

aplt.corner_anesthetic(samples=result_list[0].samples)

"""
__Wrap Up__

You've now run an end-to-end cluster lens model on a 2-main + 10-scaling + 1-halo + 2-source system.

Next steps:

- ``autolens_workspace/scripts/cluster/modeling.py``: deeper walkthrough of the same model with full
  prose on each piece.
- ``autolens_workspace/scripts/cluster/simulator.py``: how the dataset is generated end-to-end —
  including the scaling-relation truth values used here.
- ``autolens_workspace/scripts/group/features/scaling_relation/modeling.py``: galaxy-scale (extended
  imaging) counterpart of the scaling-relation tier.
- ``autolens_workspace/guides``: API reference, lensing-calculation guides, results interpretation.

**Modeling your own cluster.** Replace the dataset files in
``autolens_workspace/dataset/cluster/<name>/``:

- ``data.fits`` / ``noise_map.fits`` / ``psf.fits`` — your imaging.
- ``point_datasets.csv`` — your measured multiple-image positions, with per-source redshifts.
- ``scaling_galaxies.csv`` — your scaling-tier members' centres and luminosities.
- ``main_lens_centres.json`` / ``host_halo_centre.json`` — your individually-modelled centres.

Update ``dataset_name`` above to point at the new folder, and the rest of the script runs unchanged.
"""
