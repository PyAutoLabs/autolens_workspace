"""
Modeling: Cluster
=================

This script models the small multi-plane cluster lens simulated by ``cluster/simulator.py``: a Brightest
Cluster Galaxy (BCG) plus one satellite member at the lens redshift ``z = 0.5``, 10 lower-mass cluster
members modelled collectively via a luminosity-mass scaling relation, a standalone NFW host dark matter
halo *not* tied to any individual galaxy, and 2 background sources at *different* redshifts (``z = 1.0``
and ``z = 2.0``) which the cluster lenses into multiple images.

Cluster modeling almost always uses the *point source* API: rather than fitting the extended arc light
of each lensed source, only the image-plane positions of the brightest pixels of each multiple image are
fitted. Per-source positions, noise maps, and redshifts are loaded from a single hand-editable CSV that
the simulator writes (``point_datasets.csv``) via ``al.list_from_csv``. Cluster-member centres and
luminosities for the scaling tier are loaded from a second CSV (``scaling_galaxies.csv``) via
``al.galaxy_table_from_csv``.

__Contents__

- **Example:** What this script fits and the underlying simulator.
- **Simulation:** Overview of how the simulated dataset was generated.
- **Dataset:** Load the CCD image and the per-source point datasets from the combined CSV.
- **Model CSVs:** Load the named-galaxy mass + point CSVs written by the simulator.
- **Scaling Galaxies Table:** Load the scaling-tier centres + luminosities from the CSV.
- **Point Solver:** Set up the image-plane multiple-image solver.
- **Chi Squared:** Why this script uses an image-plane chi-squared.
- **Cluster Components:** The four categories of lensing object — main lens galaxies, scaling members, host halo, sources.
- **Redshifts:** Multi-plane redshift handling and the source-redshift / dataset-redshift pairing.
- **Model:** Compose the lens model fitted to the data.
- **Scaling Relation:** The shared two-parameter relation that ties every scaling member's mass to its luminosity.
- **Name Pairing:** How each ``Point`` model component is paired to its ``PointDataset``.
- **Search:** Configure the non-linear search used to fit the model.
- **Live Visual Update:** Push the quick-update image to a live display surface.
- **Analysis:** Create the ``AnalysisPoint`` objects, one per dataset.
- **Factor Graph:** Combine per-dataset analyses into one global ``FactorGraphModel``.
- **Run Times:** Profiling the expected run time of the model-fit.
- **Output Folder Layout:** Description of the ``output`` folder structure.
- **Result:** Overview of the results of the model-fit.

__Example__

This script fits a ``PointDataset`` of a small multi-plane cluster where:

 - There are 2 main lens galaxies with ``dPIEMassSph`` total mass distributions, each with their centre
   fixed to the values written out by the simulator, free ``sigma`` and ``r_cut``, and a vanishing core
   (``r_core`` fixed at 0 — the standard convention for BCGs and members alike) [4 parameters].
 - There are 10 scaling-tier member galaxies. Each carries a ``dPIEMassSph`` mass with centre fixed,
   ``sigma`` / ``r_cut`` derived from the shared reference-anchored scaling relation and the per-member
   luminosity, and ``r_core`` fixed at 0 [1 parameter total for the entire tier].
 - There is 1 standalone ``NFWMCRLudlowSph`` host dark matter halo with its centre fixed and a free
   ``mass_at_200`` [1 parameter].
 - There are 2 source galaxies modeled as ``Point`` sources, each with its redshift pinned to the value
   in its ``PointDataset`` row [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The ``dPIEMassSph`` profile is parameterized in Lenstool's native convention — ``sigma`` (fiducial
velocity dispersion ``v_disp`` in km/s), ``r_core`` and ``r_cut`` (arcsec) — so the
fitted posterior reads like a Lenstool results table (see ``cluster/simulator.py`` __dPIE Mass Profile__
for the conventions, and ``cluster/lenstool/`` for a published-model worked example).

The defining feature of cluster modeling is the scaling tier: 10 lower-mass members are fit jointly
with a *single free parameter* (``sigma_ref``, the fiducial velocity dispersion of a galaxy at the
reference magnitude; the relation's exponents are fixed at the Faber-Jackson values, following the
Lenstool convention). Adding more members to ``scaling_galaxies.csv`` in the future does not grow the
dimensionality of parameter space.

__Simulation__

This script fits the simulated cluster dataset produced by ``autolens_workspace/*/cluster/simulator.py``.
That simulator writes:

 - ``data.fits`` / ``noise_map.fits`` / ``psf.fits`` — CCD imaging of the cluster (used for visualization).
 - ``point_datasets.csv`` — one row per observed multiple image, grouped by source ``name``, with a
   ``redshift`` column per group. Loaded here with ``al.list_from_csv``.
 - ``mass.csv`` + ``light.csv`` + ``point.csv`` — named-galaxy CSVs carrying the full truth model
   (main galaxies + host halo + sources). Loaded here with ``al.galaxy_models_from_csv``. See
   ``scripts/cluster/csv_api.py`` for the schema walkthrough.
 - ``scaling_galaxies.csv`` — one row per scaling-tier member with columns ``y, x, luminosity``. Loaded
   here with ``al.galaxy_table_from_csv``.
 - ``tracer.json`` — true ``Tracer`` (used by visualization, not modeling).
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

Load the strong lens dataset ``cluster``, which is the dataset we will use to perform lens modeling.

We begin by loading a CCD image of the dataset. Although we perform point-source modeling and will not
use the imaging data in the model-fit, it is useful to load it for visualization.

The ``pixel_scales`` define the arc-second to pixel conversion factor of the image, which for the
dataset we are using is 0.1" / pixel.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "cluster", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/cluster/simulator.py"],
        check=True,
    )

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.1)

"""
__Point Datasets__

We load the point datasets from the combined ``point_datasets.csv`` written by the simulator. This
returns a ``List[PointDataset]`` where each entry carries:

 - ``positions``: the image-plane (y, x) positions of that source's multiple images.
 - ``positions_noise_map``: the per-position positional uncertainty.
 - ``redshift``: the source redshift (different for each source — this is a multi-plane system).

The CSV is the recommended hand-editable input for cluster datasets: a user can edit positions, noise
values, or per-source redshifts directly in a spreadsheet rather than by writing Python.
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

"""
We can print each dataset's name, positions, noise, and redshift.
"""
for dataset in dataset_list:

    print("Point Dataset Info:")
    print(dataset.info)
    print(f"Redshift: {dataset.redshift}")

"""
We can plot the cluster image with each source's positions overlaid.

Cluster-scale visualization (per-source colouring, per-image-group zoom panels, kpc scale bars) is
prototyped in ``autolens_workspace_test/scripts/cluster/visualization.py``; the default
``aplt`` helpers below are sufficient for this script.
"""
aplt.plot_array(array=data, title="")

for dataset in dataset_list:
    aplt.plot_grid(
        grid=al.Grid2DIrregular(np.atleast_2d(dataset.positions)),
        title=dataset.name,
    )

"""
__Model CSVs__

The simulator writes the truth model into three family-level CSVs — ``mass.csv`` (lens + halo mass
profiles), ``light.csv`` (lens + source light profiles), ``point.csv`` (source point components) — keyed
by a ``galaxy`` column with ``profile_class`` dispatch (see ``scripts/cluster/csv_api.py`` for the full
schema walkthrough). We load the mass and point families here; the light family is not needed for
point-source modeling (light profiles do not affect the lensing).

In a real analysis these CSVs come from upstream measurement: light-profile fits to the imaging data
populate ``light.csv``; the lens-galaxy centres in ``mass.csv`` are typically pinned to the light
centres (with their values then taken as ground truth). For cluster-scale point-source modeling the
observed centres remove a large block of degenerate parameters that the multiple-image positions alone
cannot constrain.
"""
mass_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "mass.csv", family="mass"
)
point_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "point.csv", family="point"
)

"""
__Scaling Galaxies Table__

The 10 scaling-tier cluster members live in ``scaling_galaxies.csv`` — one row per member with columns
``y, x, luminosity``. ``al.galaxy_table_from_csv`` returns a typed ``GalaxyTable`` carrying:

 - ``.centres`` — a ``Grid2DIrregular`` of per-member centres (y, x).
 - ``.luminosities`` — a list of per-member luminosities.

Both arrive in the same order as the CSV rows; the model loop below zips them together. Adding more
scaling members to a real cluster amounts to extending the CSV — no Python edits required.

In a real analysis the luminosities come from a prior light-only fit (e.g. an MGE bulge fit to the
imaging data, or a SLaM ``source_lp_0`` stage). See
``scripts/group/features/scaling_relation/modeling_for_luminosities.py`` for the standalone-fit
pattern, and ``scripts/group/features/scaling_relation/modeling.py`` for the full prose discussion.
"""
scaling_galaxies_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_galaxies_centres = scaling_galaxies_table.centres
scaling_galaxies_luminosity_list = scaling_galaxies_table.luminosities

print(f"Scaling galaxies centres: {scaling_galaxies_centres}")
print(f"Scaling galaxies luminosities: {scaling_galaxies_luminosity_list}")

"""
__Point Solver__

For point-source modeling we require a ``PointSolver``, which determines the multiple images of the mass
model for a point source at (y, x) in the source plane.

It does this by ray-tracing triangles from the image plane to the source plane and checking whether each
source-plane (y, x) point lies inside the traced triangle. The method gradually refines smaller and
smaller triangles so multiple images are computed with sub-pixel precision.

The ``PointSolver`` needs an initial image-plane grid (defined below) and a ``pixel_scale_precision``
controlling resolution — smaller values are more accurate but slower; 0.001 balances the two.

Strong-lens mass models have a "central image" which is nearly always so demagnified it cannot be
observed. We discard it via ``magnification_threshold=0.1``; raise/lower this if your dataset does/does
not include a central image.

__Chi Squared__

For point-source modeling, the likelihood can be defined in the *image plane* (compare model
multiple-image positions to observed positions) or the *source plane* (collapse observed positions back
to a common source-plane location). This script uses the image-plane chi-squared via the ``PointSolver``;
see ``autolens_workspace/*/cluster/likelihood_function`` for a full walkthrough.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=1.0,  # The pixel-scale converts pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Cluster Components__

We organise the lensing objects into four distinct categories that map directly onto the simulator:

 - ``main_lens_galaxies``: The 2 individually-modelled cluster members (BCG + satellite). Each is fitted
   with a ``dPIEMassSph`` total mass profile whose centre is fixed to ``main_lens_centres[i]``.

 - ``scaling_galaxies``: The 10 scaling-tier cluster members. Each carries a ``dPIEMassSph`` mass with
   centre fixed (from the CSV), ``sigma`` / ``r_cut`` derived from the shared reference-anchored
   scaling relation (single free normalization ``sigma_ref``; exponents fixed and tied at
   ``alpha = 0.25`` for sigma and ``beta_cut = 1 + gamma - 2*alpha = 0.7`` for r_cut, with
   ``gamma = 0.2``) plus the per-member luminosity, and a vanishing unscaled core (``r_core = 0``).
   The whole tier contributes 1 free parameter to the model regardless of how many members are in
   the CSV.

 - ``host_halo``: A single standalone ``Galaxy`` carrying the cluster's ``NFWMCRLudlowSph`` dark matter
   halo. The halo is *not* tied to any individual member — it sits "on top of" the members and
   dominates the large-scale lensing.

 - ``source_galaxies``: 2 background sources, *at different redshifts* (so this is a genuine multi-plane
   lens). Each is modeled as a ``Point`` source whose redshift is pinned to the value in its
   ``PointDataset``.

The galaxy-scale analogue of the scaling-relation tier (with extended-light imaging modeling rather than
point-source) is demonstrated at
``scripts/group/features/scaling_relation/modeling.py``.

__Redshifts__

The two sources sit at *different* redshifts (``z = 1.0`` and ``z = 2.0``), so the ``Tracer`` ray-traces
through both planes when solving for the multiple images of the further source. The lens galaxies and
host halo all sit at ``z = 0.5``.

We pin each source's ``Galaxy.redshift`` to the redshift carried by its ``PointDataset`` — that's the
whole point of the CSV ``redshift`` column. Hardcoding ``redshift=1.0`` here would silently produce the
wrong multi-plane geometry.

For the host halo, ``NFWMCRLudlowSph`` requires ``redshift_object`` and ``redshift_source`` to evaluate
the Ludlow et al. (2016) concentration-mass relation. We anchor ``redshift_source`` to the *furthest*
source redshift (matching the convention used by the simulator), so the concentration is computed
against the deepest light cone in the system.

__Model__

We compose a lens model where:

 - The 2 main lens galaxies each have a ``dPIEMassSph`` mass profile with centre fixed, free
   ``sigma`` and ``r_cut``, and ``r_core`` fixed at 0 — 2 free parameters per galaxy [4 parameters].
 - The 10 scaling-tier members share a single free parameter: ``sigma_ref``, the fiducial velocity
   dispersion of a galaxy at the reference magnitude. Each member's parameters are computed as
   ``sigma_ref * (L / L_ref) ** 0.25`` and ``r_cut_ref * (L / L_ref) ** 0.7`` with the exponents fixed
   and tied; ``r_cut_ref`` (5.0") is held fixed at the simulator truth value and every member's
   ``r_core`` is fixed at 0 [1 parameter].
 - The host halo has an ``NFWMCRLudlowSph`` mass profile with centre fixed and a free ``mass_at_200``
   [1 parameter].
 - Each source has a ``Point`` model with free ``centre_0`` / ``centre_1`` priors initialised from the
   mean of that source's observed positions [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

__Scaling Relation__

The scaling relation is reference-anchored, the convention used by Lenstool and essentially every
published cluster strong-lensing analysis, in its modern (Bergamini et al. 2019) form:

    sigma_i  = sigma_ref * (L_i / L_ref) ** alpha       # alpha    = 0.25 (Faber-Jackson)
    r_cut_i  = r_cut_ref * (L_i / L_ref) ** beta_cut    # beta_cut = 1 + gamma - 2*alpha = 0.7
    r_core_i = 0                                        # vanishing core, NOT scaled

The reference luminosity ``L_ref`` is an **explicit fixed constant** (Lenstool's reference magnitude
``mag0``), *not* the maximum luminosity of the current sample. Anchoring to a fixed reference makes the
normalization invariant to which galaxies are placed in the scaling tier, and gives the single free
parameter ``sigma_ref`` a stable, interpretable meaning: the fiducial velocity dispersion (in km/s) of
a galaxy *at the reference magnitude*, for which a prior range is easy to motivate — unlike an abstract
multiplicative factor whose units depend on the (arbitrary) luminosity normalization. In a real analysis
set ``L_ref`` to the BCG magnitude (or a characteristic L*); here we use a fiducial ``L_ref = 1.0``. Only
luminosity *ratios* enter, so the CSV's luminosity units are irrelevant; magnitude catalogues convert via
``L_i / L_ref = 10 ** (0.4 * (m_ref - m_i))``.

The exponents are *fixed and tied* rather than fitted: Faber-Jackson gives sigma ∝ L^0.25, and
demanding a mass-to-light tilt M/L ∝ L^gamma for the member's total mass M ∝ sigma^2 * r_cut enforces
``2*alpha + beta_cut = 1 + gamma``, with ``gamma = 0.2`` universally fixed in the modern literature —
so ``beta_cut = 0.7``. (The older constant-M/L convention, gamma = 0 with radii ∝ L^0.5 and scaled
cores, matches the original Lenstool-era papers but is dated.) Member cores vanish: ``r_core`` is fixed
to zero — PyAutoLens's dPIE is analytic at ``r_core = 0`` — and is never scaled with luminosity. Since
the dPIE lens strength obeys b0 ∝ sigma^2, the sigma relation is equivalent to a b0 ∝ L^(2*alpha) =
L^0.5 scaling of the internal parameterization. Freeing an
exponent (or ``r_cut_ref``) is a one-line change shown in the code comment below —
useful as a systematics test, at the cost of the degeneracy between normalization and slope that the
fixed-exponent convention exists to avoid. When member velocity dispersions are available, the standard
refinement is to calibrate the exponents kinematically (Bergamini et al. 2019: sigma ∝ L^0.27-0.28 from
MUSE member kinematics, with the r_cut exponent from the fundamental plane).

The simulator's truth value is ``sigma_ref = 85.0`` km/s (at ``L_ref = 1.0``). The prior below is much
wider than the truth to give the search room.
"""
redshift_lens = 0.5
source_redshifts = [dataset.redshift for dataset in dataset_list]

# Build af.Model[Galaxy] instances from the family CSVs. Concrete CSV values
# become fixed af.Model defaults; we then promote selected params to priors
# below. This dict is keyed by galaxy name (lens_0, lens_1, host_halo,
# source_0, source_1) — the same naming convention the simulator uses.

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

# Main Lens Galaxies: free dPIE sigma / r_cut on each; centre and redshifts stay
# fixed at the CSV values, and r_core stays fixed at the CSV's 0.0 — the vanishing
# core standard for BCGs and members alike (the dPIE is analytic at r_core = 0).
# The cosmology constants H0 / Om0 are pinned (they are model *constants*, not
# parameters to sample — if left unset they would inherit the config's default
# priors and float).
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(
        lower_limit=50.0, upper_limit=600.0
    )
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=2.0, upper_limit=40.0)
    galaxy_models[name].mass.H0 = 67.66
    galaxy_models[name].mass.Om0 = 0.30966

# Host Halo: free mass_at_200; centre + redshift_object + redshift_source stay fixed.
galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

# Source Galaxies: free Point centres with GaussianPrior initialised from the
# mean of each source's observed multiple-image positions in its PointDataset.
# This deliberately ignores the truth centre stored in point.csv — in a real
# analysis you don't know the source's true source-plane position, you only
# have the image-plane positions of its multiple images.
for i, dataset in enumerate(dataset_list):
    positions = np.atleast_2d(dataset.positions)
    point_attr = getattr(galaxy_models[f"source_{i}"], f"point_{i}")
    point_attr.centre_0 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 0])), sigma=3.0
    )
    point_attr.centre_1 = af.GaussianPrior(
        mean=float(np.mean(positions[:, 1])), sigma=3.0
    )

# Scaling Tier Members (dPIEMassSph; sigma, r_core and r_cut all derived from the
# reference-anchored scaling relation).
#
# The reference luminosity is an EXPLICIT FIXED constant (Lenstool's reference
# magnitude "mag0"), NOT the maximum luminosity of the current sample — this keeps
# the normalization invariant to which galaxies are in the tier. Set it to the BCG
# magnitude in a real analysis; here it is a fiducial L* = 1.0.
reference_luminosity = 1.0

# sigma_ref is defined ONCE outside the loop — the tier's only free parameter, the
# fiducial velocity dispersion of a reference-magnitude galaxy (km/s, so the prior
# range below is physically interpretable). Every member's sigma and r_cut are
# derived by scaling the free sigma_ref or the fixed reference truncation by its
# luminosity ratio, with the exponents fixed and tied at the Bergamini+19 values
# (sigma ∝ L^0.25; r_cut ∝ L^(1 + gamma - 2*alpha) = L^0.7 with gamma = 0.2);
# r_core is fixed at 0 (vanishing core, never scaled). The entire tier therefore
# contributes 1 free parameter regardless of how many members are in
# scaling_galaxies.csv.
#
# To free an exponent as a systematics test, replace the fixed value with e.g.
# `scaling_sigma_exponent = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)` —
# every member's sigma then derives from two shared parameters (and the r_cut
# exponent follows automatically through the 1 + gamma - 2*alpha tie).

scaling_sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=200.0)
scaling_sigma_exponent = 0.25  # alpha
scaling_gamma = 0.2
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7

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

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(**galaxy_models),
    scaling_galaxies=scaling_galaxies,
)

"""
The ``info`` attribute shows the model in a readable format. This prints the main lens galaxies, the
host halo, and the source galaxies, each with its free / fixed parameters.

The ``info`` below may not display optimally on your computer screen — for example whitespace between
parameter names on the left and parameter priors on the right may break across multiple lines. The
``info_whitespace_length`` parameter in ``config/general.yaml`` controls this; reset the Jupyter
kernel after changing it.
"""
print(model.info)

"""
__Name Pairing__

Every ``PointDataset`` has a ``name`` (e.g. ``point_0``, ``point_1``). This pairs the dataset to the
``Point`` model component with the same name. Above, the ``af.Model(al.ps.Point)`` for source ``i`` is
attached to its ``af.Model(al.Galaxy)`` under the key ``point_i`` — that's what the ``**{f"point_{i}":
point}`` expansion does.

If a dataset has no matching ``Point`` in the model, that dataset is ignored. If a ``Point`` exists with
no matching dataset, **PyAutoLens** raises an error.

In multi-source cluster lenses, this name pairing is what ensures every source's positions are fitted by
the correct model component.
"""
print(model)

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus (see
``point_source/start_here.py`` for a full description).

Other data types fit their ``start_here.py`` with ``af.MultiStartProdigy``, a much faster multi-start gradient
optimizer, and reserve ``Nautilus`` for the ``modeling.py`` example where the full posterior is needed. Cluster
fits use ``Nautilus`` in both, because the lens-equation solve behind their ``AnalysisPoint`` likelihood cannot
yet be differentiated and so a gradient optimizer cannot run on it.

The folders ``autolens_workspace/*/guides/modeling/searches`` and ``customize`` give overviews of the
non-linear searches PyAutoLens supports and how to customize the fit, including the priors.

Results are output to::

    /autolens_workspace/output/cluster/simple/modeling/<unique_identifier>/

__Unique Identifier__

The ``unique_identifier`` is generated from the model, search, and dataset, so re-running with the same
configuration resumes the existing fit. Changing any of the three regenerates the identifier.

__Iterations Per Update__

Every N iterations the search prints the max-likelihood model and best-fit image. On GPU ~2500 keeps the
output cadence around once per minute; on CPU a similar cadence is reached at lower N.

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
    name="modeling",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=10000,
    live_visual_update=False,  # Set True to open a live matplotlib window (script) or refresh a Jupyter cell (notebook).
)

"""
__Analysis__

We create one ``AnalysisPoint`` per dataset. Each defines the ``log_likelihood_function`` Nautilus uses
to fit the model to that dataset's multiple-image positions.

We then wrap each analysis in an ``AnalysisFactor`` pairing it to the *shared* lens model, and combine
all factors into a single ``FactorGraphModel``. The total log likelihood is the sum of the per-dataset
log likelihoods; each dataset gets its own output subdirectory for visualization.

__JAX__

`AnalysisPoint(use_jax=True)` per-dataset; the search driver wraps the
joint likelihood in `jax.vmap(jax.jit(...))`. Cluster point-source fits
get the largest speedup from JAX on GPU (triangle refinement + multi-
plane deflection sum dominate runtime). Force NumPy with `use_jax=False`
when debugging.
"""
analysis_list = [
    al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True)
    for dataset in dataset_list
]

"""
__Analysis Factor__

Each analysis is wrapped in an ``AnalysisFactor`` paired with the shared model. The factor-graph API is
used heavily for advanced multi-dataset lens modeling — multi-wavelength imaging, joint
imaging+interferometer fits, multiple-source cluster fits like this one.
"""
analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]

"""
__Factor Graph__

All ``AnalysisFactor`` objects combine into one ``FactorGraphModel``. The per-dataset log likelihoods
are summed; results land in a unified directory with per-dataset visualization subdirs.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
Print the global model the factor graph fits.
"""
print(factor_graph.global_prior_model.info)

"""
__Run Times__

Cluster lens modeling is computationally expensive — full Nautilus runs are typically hours on CPU,
minutes on GPU. Run times scale with (a) the log-likelihood evaluation time of a single sample and
(b) the number of iterations Nautilus needs to converge.

For this 2-main + halo + 2-source model the log-likelihood evaluation is < 1 second on CPU and < 0.02 s
on GPU. A converged fit typically takes a few thousand iterations.

__Model-Fit__

Pass the factor-graph model and the factor graph itself (as the analysis) to ``search.fit``. Watch
``autolens_workspace/output`` for on-the-fly visualization while the fit runs.

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
__Output Folder Layout__

Now the fit is running you should checkout the ``autolens_workspace/output`` folder. Results are written
to hard-disk on the fly in human-readable formats — ``.json``, ``.csv``, ``.fits``, ``.png`` and plain
text — using the highest-likelihood model found so far.

Each completed fit lives at::

    output/cluster/<dataset_name>/modeling/<unique_hash>/
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
        image/                         <- FITS + PNG: imaging + point-source products
            dataset.fits               <- data, noise-map and PSF
            fit.fits                   <- model image, residuals, chi-squared map
            tracer.fits                <- per-galaxy image-plane images
            source_plane_images.fits   <- source-plane reconstructions
            positions.png              <- observed vs model multiple-image positions
            dataset.png, fit.png, tracer.png   <- visualisations
        model.info                     <- human-readable model summary
        model.results                  <- human-readable fit summary
        search.summary                 <- search run summary
        search_internal/               <- files used to resume / visualise the search
        metadata                       <- run metadata

The ``<unique_hash>`` is a 32-character identifier derived from the model, search and dataset.

__Result__

``search.fit`` on a factor-graph returns a list of ``Result`` objects, one per ``AnalysisFactor`` (i.e.
one per dataset). Each carries the same ``max_log_likelihood_instance`` (since they share the global
model) but its own per-dataset visualization and ``FitPoint`` object.

[The ``info_whitespace_length`` config setting also controls whitespace in ``result.info``.]
"""
for result in result_list:
    print(result.max_log_likelihood_instance)

    aplt.subplot_tracer(
        tracer=result.max_log_likelihood_tracer,
        grid=grid,
    )

"""
The ``Samples`` are identical across results (the model is global), so a single corner plot from the
first result is enough.
"""
aplt.corner_anesthetic(samples=result_list[0].samples)

"""
This script gives a concise overview of the basic cluster modeling API for a small multi-plane cluster.

__Data Preparation__

If you are looking to fit your own point-source cluster data, see
``autolens_workspace/*/data_preparation/point_source/README.md`` for the input-data standards.

__HowToLens__

For a deeper understanding of how lens modeling, ray-tracing, and non-linear searches actually work, see
the **HowToLens** Jupyter notebook lectures at https://github.com/PyAutoLabs/HowToLens.
"""
