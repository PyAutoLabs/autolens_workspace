"""
Start Here: Cluster
===================

Cluster-scale strong lenses are made of:

 - One or more **Brightest Cluster Galaxies (BCGs)** and bright satellites, modelled individually with
   their own light and mass profiles.
 - **Tens to hundreds of lower-mass member galaxies**, whose collective mass perturbs the deflection
   field non-trivially but whose individual contributions are too weak to constrain on their own. These
   are modelled jointly on a luminosity-mass scaling relation, so the entire population shares a single
   free parameter regardless of how many members are in the catalogue.
 - **One or more cluster-scale dark matter halos** (``10^14 – 10^15`` M_sun), modelled with NFW-like
   profiles and not tied to any individual galaxy.
 - **Multiple background sources at different redshifts**, multiply imaged by the cluster — this makes
   cluster lensing a genuine multi-plane ray-tracing problem.

This script gets you fitting a real cluster-scale lens system in roughly 15 minutes. We model **real
data**: **Abell 2744** ("Pandora's Cluster", z = 0.308), a Hubble Frontier Fields cluster and one of
the most powerful gravitational lenses known. The multiple-image positions and spectroscopic
redshifts, and the cluster-member catalogue, come from the published lens-model inputs of Bergamini
et al. 2023 (A&A 670, A60): we fit the 7 "gold" source systems (25 multiple images, sources from
``z = 1.7`` to ``z = 5.7`` — a genuinely multi-plane problem) with 2 individually-modelled BCGs,
188 scaling-tier members, and an NFW host halo.

For galaxy-scale lenses (a single dominant lens and a single source), start with
``imaging/start_here.ipynb`` instead; for 2+ co-dominant lens galaxies with no host halo see
``multi_galaxy/start_here.ipynb``, and for group-scale systems (optional host halo, one extended source)
see ``group/start_here.ipynb``.

__Contents__

- **JAX:** GPU/CPU acceleration; cluster fits take ~10 minutes on a GPU.
- **Capabilities:** What cluster modeling supports, and practical tips for using it.
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
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Result:** Inspect the maximum-likelihood model.
- **Wrap Up:** Where to go next.

__JAX__

PyAutoLens runs cluster point-source fits on JAX by default —
`al.AnalysisPoint(use_jax=True)` (auto-enabled) routes the likelihood
through `jax.vmap(jax.jit(...))`. Cluster fits benefit the most from
GPU acceleration: the multi-galaxy multi-plane deflection sum + the
`PointSolver` triangle refinement loop are the dominant costs and
both vectorise cleanly on GPU. Expect ~10 minutes per fit on GPU vs
30+ on CPU.

For the broader JAX principles, see the top-level
`autolens_workspace/start_here.py` `__JAX__` section. The
`scripts/cluster/simulator.py` `__JAX JIT — Point Solver__` section
shows the post-Phase-2 `PointSolver(use_jax=True)` +
`autolens.jax.register_tracer_classes(tracer)` pattern in action.

__Capabilities__

Cluster modeling with **PyAutoLens** offers:

 - JAX-accelerated image-plane chi-squared, over 50× faster than mainstream cluster modeling tools.
 - Multi-plane ray tracing of arbitrary complexity, supported natively.
 - Hand-editable CSV inputs (point datasets, scaling-galaxy catalogues) that make iterating on a real
   cluster straightforward.

Practical tip: the default ``aplt`` visualization is tuned for galaxy-scale lenses, so when inspecting
cluster fits you may prefer to build custom figures of the multiple-image positions and source planes.

__Google Colab Setup__

The ``start_here`` examples are runnable on Google Colab without local PyAutoLens installation. The
block below installs the dependencies and downloads the example dataset if you're on Colab; running it
locally is a no-op.
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
    raise_error_if_not_gpu=False  # Switch to True to require GPU on Colab.
)

"""
__Imports__
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

We load the real Abell 2744 dataset. The dataset folder contains:

 - ``point_datasets.csv`` — one row per observed multiple image, grouped by source ``name``, with a
   ``redshift`` column per source. These are the Bergamini et al. 2023 gold systems: 25 images with
   spectroscopic redshifts, as (y, x) arc-second offsets about the projected cluster core (the same
   centre used by the weak-lensing examples in ``scripts/weak/``, so strong- and weak-lensing
   constraints on this cluster share a coordinate frame).
 - ``scaling_galaxies.csv`` — one row per scaling-tier member with columns ``y, x, luminosity``:
   188 cluster members from the same paper's catalogue, with luminosities relative to the BCG
   (from their F160W magnitudes).
 - ``mass.csv`` / ``point.csv`` — named-galaxy CSVs defining the individually-modelled galaxies:
   the two brightest core members (the BCGs) as dPIE profiles, the NFW host halo centred on the
   BCG, and one point source per system (see ``csv_api.py`` for the schema, and the dataset
   folder's ``README.md`` + ``prep.py`` for full provenance).

For visualization we also download an HST H-band image cutout of the cluster from the CDS
hips2fits service on the first run (~1.4 MB, cached). The image is only used for plotting — the
model is fitted to the multiple-image positions.
"""
dataset_name = "a2744"
dataset_path = Path("dataset") / "cluster" / dataset_name

data_fits_path = dataset_path / "data.fits"

HIPS2FITS_URL = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    "?hips=CDS%2FP%2FHST%2FH&ra=3.5875&dec=-30.3972"
    "&width=600&height=600&fov=0.05&projection=TAN&format=fits"
)

if not data_fits_path.exists():
    import urllib.request

    print(
        "Downloading HST H-band image of Abell 2744 for visualization (one-off, ~1.4 MB) ..."
    )
    try:
        # An explicit timeout is essential: `urlretrieve` has none, so a slow or
        # stalled hips2fits response hangs the script indefinitely (in CI this
        # blocked until the build-script timeout). On any failure we continue —
        # the image is used for visualization only.
        with urllib.request.urlopen(HIPS2FITS_URL, timeout=30) as response:
            data_fits_path.write_bytes(response.read())
    except Exception as e:
        print(
            f"Image download failed ({e}) — continuing without it (visualization only)."
        )

if data_fits_path.exists():
    # hips2fits returns 0.3"/pixel for this 0.05 deg / 600 pixel cutout.
    data = al.Array2D.from_fits(file_path=data_fits_path, pixel_scales=0.3)

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
grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=1.0)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Cluster Components__

The model has four tiers, one per cluster component:

 - **Main lens galaxies (2):** the two brightest core members (the BCG region galaxies),
   individually-modelled ``dPIEMassSph`` profiles with centre fixed to the observed light centres,
   free ``sigma`` and ``r_cut``, and a vanishing core (``r_core`` fixed at 0 — the standard convention
   for BCGs and members alike; PyAutoLens's dPIE is analytic at ``r_core = 0``). These are Lenstool's
   native dPIE parameters, so the posterior reads like a Lenstool results table.
   **4 free parameters total.**

 - **Scaling-tier members (188):** ``dPIEMassSph`` profiles with centre fixed to the CSV centres.
   ``sigma`` and ``r_cut`` derive from the reference-anchored relation used by Lenstool and standard in
   published cluster analyses, in its modern (Bergamini et al. 2019) form:
   ``sigma = sigma_ref * (L / L_ref) ** alpha`` with ``alpha = 0.25`` (Faber-Jackson) and
   ``r_cut = r_cut_ref * (L / L_ref) ** beta_cut`` with ``beta_cut = 1 + gamma - 2*alpha = 0.7``
   (``gamma = 0.2``, the universally fixed mass-to-light tilt M/L ∝ L^gamma); every member's ``r_core``
   is fixed at 0 and never scaled. ``L_ref`` is an explicit fixed reference luminosity (Lenstool's
   ``mag0``), *not* the sample max — only the normalization ``sigma_ref``, the fiducial velocity
   dispersion of a reference-magnitude galaxy, is fitted. Our member luminosities are normalised to the
   BCG's F160W flux, so ``L_ref = 1.0`` anchors the relation to the BCG itself.
   **1 free parameter total for the whole tier — independent of the number of members.**

 - **Host dark matter halo:** a standalone ``Galaxy`` carrying an ``NFWMCRLudlowSph`` halo with
   centre fixed on the BCG and a free ``mass_at_200``. (Abell 2744 is a merging cluster — published
   models use several halos; one halo is the deliberately simple starting point, and adding a second
   is a CSV-level edit.) **1 free parameter.**

 - **Source galaxies (7):** ``Point`` models, redshift pinned to each source's per-dataset
   spectroscopic value, with ``GaussianPrior`` centre priors initialised from the mean of each
   source's observed positions. **14 free parameters total.**

**Total: N = 20 free parameters.** Adding more rows to ``scaling_galaxies.csv`` does not grow N — that's
the defining feature of cluster-scale modeling on a scaling relation. See
``scripts/cluster/modeling.py`` for the full prose on the scaling-relation convention (why the
normalization anchors to a reference galaxy, why the exponent is fixed, and the kinematic calibrations
that refine it).

__Redshifts__

The seven sources sit at different spectroscopic redshifts (``z = 1.688`` to ``z = 5.662``); the
``Tracer`` automatically ray-traces through every source plane when solving the further sources.
Lens galaxies (main + scaling) and the host halo all sit at the cluster redshift ``z = 0.308``.
``NFWMCRLudlowSph`` needs ``redshift_source`` to evaluate the Ludlow et al. (2016)
concentration-mass relation — we anchor it to the *furthest* source.

__Model__

The model is composed below in four blocks: main-tier loop, host halo, source-tier loop, scaling-tier
loop (defining the shared ``sigma_ref`` normalization once outside the loop). The four blocks are then
bundled into a single ``af.Collection`` model that the analysis will receive.
"""
redshift_lens = 0.308
source_redshifts = [dataset.redshift for dataset in dataset_list]

# Build af.Model[Galaxy] instances directly from the family CSVs. Concrete CSV
# values become fixed af.Model defaults; we then promote selected params to
# priors below. Keys: lens_0, lens_1, host_halo, source_0, source_1.

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

# Main Lens Galaxies: free dPIE sigma / r_cut; centre and redshifts stay fixed at
# the CSV values, and r_core stays fixed at the CSV's 0.0 — the vanishing core
# standard for BCGs and members alike (the dPIE is analytic at r_core = 0). The
# cosmology constants H0 / Om0 are pinned (they are model *constants*, not
# parameters to sample — if left unset they would inherit the config's default
# priors and float).
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(
        lower_limit=50.0, upper_limit=600.0
    )
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=2.0, upper_limit=40.0)
    galaxy_models[name].mass.H0 = 67.66
    galaxy_models[name].mass.Om0 = 0.30966

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

# Scaling Tier (reference-anchored: sigma_ref is the single shared free parameter,
# the fiducial velocity dispersion of a galaxy at the reference magnitude, in km/s;
# per-member sigma and r_cut derive from it with the exponents fixed and tied at
# the Bergamini+19 values (sigma ∝ L^0.25; r_cut ∝ L^(1 + gamma - 2*alpha) = L^0.7
# with gamma = 0.2); r_core is fixed at 0 (vanishing core, never scaled). The
# reference luminosity is an EXPLICIT FIXED constant (Lenstool's "mag0"); our
# member luminosities are normalised to the BCG's F160W flux, so L_ref = 1.0
# anchors the relation to the BCG).

scaling_sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
scaling_sigma_exponent = 0.25  # alpha
scaling_gamma = 0.2
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7

reference_luminosity = 1.0
scaling_r_core_fixed = 0.0
scaling_r_cut_ref_fixed = 5.0

scaling_galaxies_list = []
for centre, luminosity in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosity_list
):
    luminosity_ratio = luminosity / reference_luminosity

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.sigma = scaling_sigma_ref * luminosity_ratio**scaling_sigma_exponent
    mass.r_core = scaling_r_core_fixed
    mass.r_cut = scaling_r_cut_ref_fixed * luminosity_ratio**scaling_rcut_exponent
    mass.redshift_object = redshift_lens
    mass.redshift_source = max(source_redshifts)
    mass.H0 = 67.66
    mass.Om0 = 0.30966

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

We use Nautilus, a robust nested-sampling algorithm. ``n_live=150`` is a sensible default for a 22-D
model — increase it for more complex clusters. ``n_batch=50`` batches the GPU log-likelihood
evaluations for throughput.

__Why Not MultiStartProdigy?__

The imaging and interferometer ``start_here.py`` examples fit with ``af.MultiStartProdigy``, a much
faster multi-start gradient optimizer. Cluster fits cannot use it yet: this is an ``AnalysisPoint``
likelihood, computed by solving the lens equation for each system's multiple images, and PyAutoLens
cannot yet differentiate through that solve. Nautilus also gives us the posterior that the corner
plot at the end of this script is built from.

Results are written to ``autolens_workspace/output/cluster/a2744/start_here/<unique_hash>/``. The
``unique_hash`` is generated from the model, search settings, and dataset — re-running with the same
configuration resumes the existing fit.

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
    path_prefix=Path("cluster"),
    name="start_here",
    unique_tag=dataset_name,
    n_live=150,
    n_batch=50,
    iterations_per_quick_update=2500,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
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

You've now run an end-to-end cluster lens model on real data: Abell 2744 with 2 BCGs, 188
scaling-tier members, an NFW host halo and 7 spectroscopically-confirmed source systems.

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
- ``mass.csv`` / ``point.csv`` — your individually-modelled galaxies (centres and profiles), in the
  named-galaxy CSV schema (see ``csv_api.py``).

Update ``dataset_name`` above to point at the new folder, and the rest of the script runs unchanged.
"""
