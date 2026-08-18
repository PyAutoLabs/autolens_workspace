"""
Log Likelihood Function: Cluster Point Source
=============================================

This script provides a step-by-step guide of the cluster point-source ``log_likelihood_function``,
the figure-of-merit Nautilus optimises when fitting a cluster lens model to ``point_datasets.csv``.

Cluster point-source modelling has two distinct likelihood flavours:

 1. **Source-plane chi²** (``FitPositionsSource``) — ray-trace every observed image-plane position
    back through the lens, and measure the magnification-weighted scatter of the back-traced
    positions around the source-plane centre. Cheap (no forward solve) and JAX-friendly.

 2. **Image-plane chi²** (``FitPositionsImagePair``) — forward-solve the model's image-plane
    positions for each source via the ``PointSolver``, pair each model position to the closest
    observed position, and measure the image-plane residuals. More intuitive (residuals in
    arc-seconds), but slower and carries pairing pathologies the source-plane variant avoids.

Each flavour also has a ``*Solved`` sibling (``FitPositionsSourceSolved`` and the image-plane
solved variants) which solves the source-plane centre analytically instead of sampling it — see
the ``Source-Plane Centroid`` section below and ``guides/point_source_pairing.py``.

We walk through both, end to end, with the actual library formulae. The standard cluster model is
assumed: all lens-plane galaxies at ``z = 0.5``, two background sources at *different* redshifts
(``z = 1.0`` and ``z = 2.0``). Multi-plane ray tracing therefore applies and we explain how the
recursive lens equation handles it.

This is a companion to ``scripts/imaging/likelihood_function.py`` and
``scripts/group/likelihood_function.py``, both of which cover extended-imaging likelihoods. The
cluster surface is the first one in the workspace tutorial set that uses *point sources*
exclusively + multi-plane ray tracing + many lens galaxies in the deflection sum.

__Aims__

 - Provide a resource that authors can include in papers using cluster modelling, so readers can
   understand the likelihood (including references to the prior literature it builds on) without
   needing to re-derive equations or trace the PyAutoLens source.
 - Make the difference between source-plane and image-plane chi² concrete so users can decide
   which flavour fits their dataset + compute budget.
 - Document the multi-plane ray-tracing convention used at cluster scale.

__Contents__

- **JAX:** Model-fits run this likelihood function through JAX — see ``scripts/guides/using_jax.py``.
- **Dataset:** Load the CCD imaging (used only for visualisation) and the per-source point datasets.
- **Truth Model:** Load the truth model from the family CSVs (mass / point / scaling_galaxies).
- **Tracer:** Build the ``Tracer`` carrying every lens-plane galaxy + the two source-plane galaxies.

- **Source-Plane Chi Squared: Concept:** Why "back-trace the observed images" is a likelihood.
- **Multi-Plane Ray Tracing:** Recursive lens equation; scaling factors per source.
- **Back-Traced Source-Plane Positions:** Computing source-plane positions per multiple image.
- **Source-Plane Centroid:** Truth Point centre vs barycenter (which to use when).
- **Residual Map:** Source-plane distance per multiple image.
- **Magnifications at Positions:** Hessian-derived magnification per image.
- **Chi Squared Map (Source):** ``residual² × magnification² / noise²``.
- **Per-Source Chi Squared:** Sum within one source's multiple-image set.
- **Total Chi Squared (Source):** Sum across sources.
- **Noise Normalization (Source):** ``sum log(2π × magnification^-2 × noise²)``.
- **Source-Plane Log Likelihood:** ``-0.5 × (chi² + noise_normalization)``.
- **Source-Plane Validation:** ``al.FitPositionsSource`` matches the step-by-step result.

- **Image-Plane Chi Squared: Concept:** Forward-solve, pair, measure.
- **Point Solver Setup:** Starting grid, precision, magnification threshold.
- **Forward Solving Model Positions:** Per source, via the multi-plane-aware solver.
- **Pairing Model to Observed:** Three pairing schemes + the too-many / too-few pathology.
- **Image-Plane Residual Map:** Image-plane distances per pair.
- **Chi Squared Map (Image):** ``residual² / noise²`` — no magnification weighting here.
- **Per-Source / Total Chi Squared:** Sum across pairs and sources.
- **Noise Normalization (Image):** ``sum log(2π × noise²)``.
- **Image-Plane Log Likelihood:** ``-0.5 × (chi² + noise_normalization)``.
- **Image-Plane Validation:** ``al.FitPositionsImagePair`` matches the step-by-step result.

- **Source-Plane vs Image-Plane: When to Use Which:** Practical comparison.
- **Wrap Up:** Pointers to ``modeling.py``, ``csv_api.py``, and the test-workspace sanity diagnostic.

__JAX__

Model-fits evaluate this likelihood function through JAX rather than NumPy, which is what makes cluster
modelling fast — ``scripts/guides/using_jax.py`` shows a likelihood function JAX-compiled via both the
``Analysis`` object and the ``Fitness`` object a non-linear search drives.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import subprocess
import sys
from pathlib import Path

import numpy as np

import autogalaxy as ag
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The cluster point-source dataset lives in ``dataset/cluster/simple/``. The CCD image (``data.fits``)
is loaded for visualisation only — point-source modelling does NOT fit the imaging directly. The
positions of the multiple images of each source, plus per-position uncertainties, drive the
likelihood.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "cluster" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.

This script reads the truth model CSVs as well as the image, so it also checks ``mass.csv`` — a
directory left half-written by an interrupted simulator run must still trigger a rebuild.
``should_simulate`` is evaluated first so that its ``PYAUTO_SMALL_DATASETS`` rebuild always runs.
"""
if (
    al.util.dataset.should_simulate(str(dataset_path))
    or not (dataset_path / "mass.csv").exists()
):
    subprocess.run([sys.executable, "scripts/cluster/simulator.py"], check=True)

data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.1)

"""
__Point Datasets__

``point_datasets.csv`` carries one row per observed multiple image, grouped by source ``name``.
``al.list_from_csv`` returns a ``List[PointDataset]`` where each entry carries:

 - ``name`` — source identifier (e.g. ``point_0``), used for name pairing with the model.
 - ``positions`` — image-plane (y, x) positions of every multiple image of that source.
 - ``positions_noise_map`` — per-position positional uncertainty in arc-seconds.
 - ``redshift`` — the source redshift (different per source — this is a multi-plane system).
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

for dataset in dataset_list:
    print(f"  {dataset.name}: z={dataset.redshift}  {len(dataset.positions)} images")


"""
__Truth Model__

The simulator wrote the truth lens model into the family CSVs (``mass.csv`` for dPIE + NFW mass
profiles; ``point.csv`` for source-galaxy ``Point`` components). We load both and build the
truth ``Tracer``. The scaling tier (10 low-mass members) has its own legacy ``scaling_galaxies.csv``
schema with shared scaling-relation parameters.

The likelihood walkthrough below evaluates the chi² at the *truth* model — the chi² ought to be at
or near its minimum here.
"""
mass_table = al.galaxy_models_from_csv(dataset_path / "mass.csv", family="mass")
point_table = al.galaxy_models_from_csv(dataset_path / "point.csv", family="point")
scaling_table = al.galaxy_table_from_csv(dataset_path / "scaling_galaxies.csv")

galaxies_by_name = al.galaxies_from_csv_tables(mass_table, point_table)

redshift_lens = 0.5
source_redshifts = sorted({float(d.redshift) for d in dataset_list})

main_lens_galaxies = [galaxies_by_name["lens_0"], galaxies_by_name["lens_1"]]
host_halo_galaxy = galaxies_by_name["host_halo"]
source_galaxies = [galaxies_by_name["source_0"], galaxies_by_name["source_1"]]

"""
The scaling tier builds a dPIE per member from the legacy CSV and the reference-anchored scaling
relation in its modern (Bergamini et al. 2019) form — ``sigma ∝ L^0.25`` and
``r_cut ∝ L^(1 + gamma - 2*alpha) = L^0.7`` with ``gamma = 0.2``, vanishing unscaled cores (see
``modeling.py`` for the full rationale). ``REFERENCE_LUMINOSITY`` is an explicit fixed constant
(Lenstool's ``mag0``), not the sample max, and the constants below match the simulator truth so
members are reproduced exactly.
"""
scaling_galaxies = []
SCALING_SIGMA_REF_TRUTH = 85.0
SCALING_SIGMA_EXPONENT = 0.25  # alpha
SCALING_GAMMA = 0.2
SCALING_RCUT_EXPONENT = 1.0 + SCALING_GAMMA - 2.0 * SCALING_SIGMA_EXPONENT  # 0.7
SCALING_R_CORE = 0.0  # vanishing core — fixed, never scaled
SCALING_R_CUT_REF = 5.0
REFERENCE_LUMINOSITY = 1.0
for centre, luminosity in zip(
    scaling_table.centres.in_list, scaling_table.luminosities
):
    luminosity_ratio = luminosity / REFERENCE_LUMINOSITY
    scaling_galaxies.append(
        al.Galaxy(
            redshift=redshift_lens,
            mass=al.mp.dPIEMassSph(
                centre=tuple(centre),
                sigma=SCALING_SIGMA_REF_TRUTH
                * luminosity_ratio**SCALING_SIGMA_EXPONENT,
                r_core=SCALING_R_CORE,
                r_cut=SCALING_R_CUT_REF * luminosity_ratio**SCALING_RCUT_EXPONENT,
                redshift_object=redshift_lens,
                redshift_source=max(source_redshifts),
            ),
        )
    )

"""
__Tracer__

The tracer carries:

 - 2 main lens galaxies (BCG + satellite) — individually-modelled dPIE mass profiles.
 - 10 scaling-tier member galaxies — dPIE mass profiles whose ``sigma`` and ``r_cut`` derive from the
   reference-anchored scaling relation ``sigma = sigma_ref × (L/L_ref)^0.25``,
   ``r_cut ∝ (L/L_ref)^0.7`` (the Bergamini+19 tied-exponent convention), with vanishing cores.
 - 1 host dark matter halo — ``NFWMCRLudlowSph`` at the cluster centre.
 - 2 source galaxies — ``Point`` profiles at distinct redshifts (multi-plane).

PyAutoLens automatically groups galaxies by redshift into ``planes`` and ray-traces through them
in redshift order. For this cluster the planes are: ``z=0.5`` (lens-plane, holds 13 galaxies),
``z=1.0`` (source_0 plane), ``z=2.0`` (source_1 plane).
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies
    + scaling_galaxies
    + [host_halo_galaxy]
    + source_galaxies
)

print(
    f"Tracer has {len(tracer.planes)} planes at redshifts "
    f"{[float(p.redshift) for p in tracer.planes]}"
)


"""
__Source-Plane Chi Squared: Concept__

If the lens model is correct, every observed image position of a given source ought to ray-trace
back to (approximately) the same source-plane location: the source's true centre. The figure of
merit for the **source-plane chi²** is the scatter of those back-traced positions around the
source-plane reference, weighted by magnification (so source-plane residuals are converted back to
image-plane scale where the noise is defined).

This contrasts with the more intuitive image-plane chi² which compares observed image positions to
the model's *forward-solved* multiple-image positions. Source-plane chi² is cheaper (no forward
solve required, just one back-projection per image), and the math is JAX-friendly. The trade-off
is that the magnification weighting amplifies any positional error in the back-projection: at
cluster scale where magnifications near multi-image positions are ~100×, even sub-arcsecond
source-plane residuals can drive chi² to 10⁶–10⁸ at the truth.

This is the figure of merit used by ``al.FitPositionsSource`` and is described in detail in
Jullo et al. 2007 ("A Bayesian approach to strong lensing modelling of galaxy clusters") and the
references therein.

__Multi-Plane Ray Tracing__

The cluster has two sources at *different* redshifts (``z = 1.0`` and ``z = 2.0``), so ray-tracing
each source's positions back to its source plane goes through every earlier plane along the way.
This is **multi-plane ray tracing**, formalised by the recursive lens equation:

.. math::

    \\theta_{j} = \\theta_{0} - \\sum_{i=1}^{j-1} \\beta_{ij} \\alpha_{i}(\\theta_{i})

where:

 - :math:`\\theta_{0}` is the image-plane position,
 - :math:`\\theta_{i}` is the position of the ray at plane ``i``,
 - :math:`\\alpha_{i}(\\theta_{i})` is the deflection at plane ``i`` (sum of every galaxy at that
   plane's redshift),
 - :math:`\\beta_{ij} = (D_{ij} \\, D_{s}) / (D_{j} \\, D_{is})` is a cosmological scaling factor
   that scales the deflection from plane ``i``'s angular-diameter convention to plane ``j``'s.

For our cluster the relevant cases per source are:

 - **Source 0 (z=1.0)** — back-tracing its image positions just hits the lens plane (z=0.5).
   Two planes total, so the scaling factor reduces to 1.0 and the equation is the familiar
   :math:`\\theta_{1} = \\theta_{0} - \\alpha_{0}(\\theta_{0})`.

 - **Source 1 (z=2.0)** — back-tracing hits the lens plane (z=0.5) AND the source_0 plane
   (z=1.0). Even though source_0 carries no mass, the recursive equation still walks through its
   plane; the actual contribution depends on whether any galaxy at z=1.0 has a non-zero mass
   profile (none here — the source_0 ``Point`` profile contributes nothing to the deflection
   field).

The library encapsulates all of this in ``Tracer.deflections_between_planes_from(grid, plane_i=0,
plane_j=<source's plane index>)``. The function walks the planes from ``plane_i`` to ``plane_j``
in redshift order, applying the per-plane :math:`\\alpha_{i}(\\theta_{i})` sums and the scaling
factors :math:`\\beta_{ij}` automatically. The full derivation of the recursive equation, the
sign conventions, and the cosmological scaling factors live in the multi-plane guide at
``autolens_workspace/scripts/guides/advanced/multi_plane.py``.

In practice the easiest entry point isn't ``deflections_between_planes_from`` (which returns the
*differences* between plane positions and requires you to apply the lens equation manually) but
``Tracer.traced_grid_2d_list_from(grid)``. That function returns the list of grids per plane,
fully traced through the recursive lens equation, with the cosmological scaling factors already
applied. For our 3-plane cluster:

 - ``traced_grids[0]`` is the input image-plane grid (unchanged).
 - ``traced_grids[1]`` is the source_0 plane (z=1.0) position of every input ray after passing
   through the lens-plane deflection.
 - ``traced_grids[2]`` is the source_1 plane (z=2.0) position of every input ray after passing
   through the lens plane *and* the source_0 plane (the recursive step).

Each source's back-traced positions are then just ``traced_grids[<source's plane index>]``.
"""
source_plane_positions_per_source = []
for i, dataset in enumerate(dataset_list):
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
    traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.positions)
    source_plane_positions_per_source.append(traced_grids[plane_index])

    print(
        f"  {dataset.name}: plane_index={plane_index}, "
        f"back-traced positions = {traced_grids[plane_index].in_list}"
    )


"""
__Back-Traced Source-Plane Positions: Conceptual Recap__

The traced grid at plane ``j`` is, by construction:

.. math::

    \\theta_{j} = \\theta_{0} - \\alpha_{\\text{multi-plane}}(\\theta_{0}; j)

where the multi-plane deflection :math:`\\alpha_{\\text{multi-plane}}` is the recursive sum
discussed in the previous section — exactly the result you would get from manually applying
``deflections_between_planes_from`` and then ``grid_2d_via_deflection_grid_from``. Using
``traced_grid_2d_list_from`` is the same thing in one call.

__Source-Plane Centroid__

The "reference point" against which we measure the source-plane scatter has two options:

 1. **Free ``Point`` centre.** Each source carries a ``Point`` profile in the model whose
    ``centre`` is a free parameter (or fixed to the truth here). At a model fit, ``Point.centre``
    *is* the source-plane (y, x) the multiple images should converge to, and the sampler explores
    it as two non-linear parameters per source.

 2. **Analytically solved centre.** Give each source the parameter-free ``al.ps.PointSolved``
    profile and fit with ``al.FitPositionsSourceSolved``: the centre is computed in closed form as
    the precision-weighted mean of the back-traced positions (Lombardi 2024, arXiv:2406.15280,
    §5.1), with a tensor weighting derived from the lensing Jacobian, and the likelihood is
    analytically marginalized over it. This removes 2 free parameters per source from the search —
    at cluster scale, with many sources, a substantial dimensionality reduction. See
    ``guides/point_source_pairing.py`` for the full solved-variant matrix.

Option (1) is what ``al.FitPositionsSource(profile=point_profile)`` uses and is what this
walkthrough demonstrates, so the residuals have a well-defined physical meaning (distance from
truth). Option (2) is its solved sibling ``FitPositionsSourceSolved``, whose reference point is a
weighted barycenter of the back-traced positions rather than a sampled parameter.
"""
source_plane_centroids = []
for i, dataset in enumerate(dataset_list):
    # source_i's Point profile is attached to source_galaxies[i] under attr name "point_i".
    point_profile = getattr(source_galaxies[i], dataset.name)
    source_plane_centroids.append(point_profile.centre)


"""
__Residual Map__

The per-image residual is just the source-plane distance between each back-traced position and the
source-plane centroid:

.. math::

    r_{i} = |\\theta_{j,i} - \\theta_{j,\\text{centre}}|
"""
residuals_per_source = []
for i, dataset in enumerate(dataset_list):
    sp_positions = source_plane_positions_per_source[i]
    centre = source_plane_centroids[i]
    residuals = sp_positions.distances_to_coordinate_from(coordinate=centre)
    residuals_per_source.append(residuals)

    print(f"  {dataset.name}: residuals = {np.asarray(residuals)}")


"""
__Magnifications at Positions__

The source-plane chi² weights each residual by the local **magnification** at that image position.
Why: position noise is defined in the image plane (arcsec), but the residual is in the source
plane. The magnification converts a source-plane distance back to its image-plane scale.
Specifically the formula below uses magnification squared as a multiplicative weight.

PyAutoLens computes magnification from the Hessian of the deflection field via
``Tracer.magnification_2d_via_hessian_from(grid)``. Physically the Hessian gives the linear
distortion matrix at each image position; its determinant is the magnification (signed: negative
for parity-flipped images, but ``magnification_2d_via_hessian_from`` returns the absolute value).
The conceptual chain is **deflection field → Hessian via finite-difference → eigenvalues
(convergence κ and shear γ) → magnification :math:`\\mu = 1 / |(1-\\kappa)^2 - \\gamma^2|`**.

Full derivation and the eigenvalue / critical-curve discussion live in the lens-calc guide at
``autolens_workspace/scripts/guides/lens_calc.py``.

For multi-plane lenses the function uses the full multi-plane deflection in the Hessian, so the
magnification correctly reflects the cumulative distortion through every intervening plane. The
``LensCalc`` helper bundles the Hessian, magnification, and critical-curve calculations; we build
one per source so each magnification is evaluated against the correct multi-plane chain (the
source's plane index).

The library wraps this with an ``abs(...)`` so the returned magnification is always positive (raw
magnification can be negative when the image is parity-flipped; squaring it in the chi² formula
makes the sign irrelevant anyway).
"""
magnifications_per_source = []
for i, dataset in enumerate(dataset_list):
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
    od = ag.LensCalc.from_tracer(
        tracer=tracer, use_multi_plane=True, plane_j=plane_index
    )
    mag = abs(od.magnification_2d_via_hessian_from(grid=dataset.positions))
    magnifications_per_source.append(mag)

    print(f"  {dataset.name}: magnifications = {np.asarray(mag)}")


"""
__Chi Squared Map (Source)__

The per-image source-plane chi² combines the residual, the magnification, and the position noise:

.. math::

    \\chi^{2}_{i} = \\frac{r_{i}^{2} \\, \\mu_{i}^{2}}{\\sigma_{i}^{2}}

This is the formula in ``autolens/point/fit/positions/source/separations.py`` ::

    chi_squared_map = residual_map**2 / (magnifications_at_positions.array**-2 * noise_map.array**2)

which expands to ``residual² × magnification² / noise²``. (The :math:`\\mu^{-2}` in the denominator
of the source-code form cancels the :math:`\\mu^{2}` in the numerator and leaves the form above.)
"""
chi_squared_maps_per_source = []
for i, dataset in enumerate(dataset_list):
    r = np.asarray(residuals_per_source[i])
    mu = np.asarray(magnifications_per_source[i])
    sigma = np.asarray(dataset.positions_noise_map)
    chi_sq_map = r**2 * mu**2 / sigma**2
    chi_squared_maps_per_source.append(chi_sq_map)

    print(f"  {dataset.name}: chi² per image = {chi_sq_map}")


"""
__Per-Source Chi Squared__

Per-source: sum the chi² map across that source's multiple images.
"""
chi_squared_per_source = [float(np.sum(m)) for m in chi_squared_maps_per_source]
for i, dataset in enumerate(dataset_list):
    print(f"  {dataset.name}: chi² = {chi_squared_per_source[i]:.4e}")


"""
__Total Chi Squared (Source)__

Each source is independent (different redshift, different multiple-image set), so the total chi²
is just the sum.
"""
total_chi_squared_source = float(sum(chi_squared_per_source))
print(f"Total source-plane chi² = {total_chi_squared_source:.4e}")


"""
__Noise Normalization (Source)__

The full Gaussian log-likelihood carries a normalisation term that depends on the noise:

.. math::

    \\mathcal{N} = \\sum_{i} \\log \\left( 2\\pi \\, \\mu_{i}^{-2} \\, \\sigma_{i}^{2} \\right)

The magnification factor appears here too because the chi² formula effectively treats each image
position as having an *effective* source-plane noise of :math:`\\sigma_{i} / \\mu_{i}`. The
normalisation matches that interpretation so the resulting expression is a well-defined Gaussian
log-likelihood.
"""
noise_normalizations_per_source = []
for i, dataset in enumerate(dataset_list):
    mu = np.asarray(magnifications_per_source[i])
    sigma = np.asarray(dataset.positions_noise_map)
    nn = float(np.sum(np.log(2 * np.pi * (mu**-2) * sigma**2)))
    noise_normalizations_per_source.append(nn)

total_noise_normalization_source = float(sum(noise_normalizations_per_source))
print(
    f"Total source-plane noise normalization = {total_noise_normalization_source:.4e}"
)


"""
__Source-Plane Log Likelihood__

The standard Gaussian log-likelihood:

.. math::

    \\log L = -\\frac{1}{2} \\left( \\chi^{2} + \\mathcal{N} \\right)

This is what Nautilus maximises during a cluster point-source model fit. The chi² term encodes
goodness-of-fit; the normalisation absorbs the constant noise-dependent prefactor of the Gaussian.
"""
log_likelihood_source = -0.5 * (
    total_chi_squared_source + total_noise_normalization_source
)
print(f"Source-plane log likelihood = {log_likelihood_source:.4e}")


"""
__Source-Plane Validation__

To verify the step-by-step derivation, we instantiate ``al.FitPositionsSource`` and confirm its
``log_likelihood`` matches. ``FitPositionsSource`` does exactly the calculation above internally
(see ``autolens/point/fit/positions/source/separations.py``).
"""
sum_library_log_likelihood_source = 0.0
for i, dataset in enumerate(dataset_list):
    point_profile = getattr(source_galaxies[i], dataset.name)
    fit = al.FitPositionsSource(
        name=dataset.name,
        data=dataset.positions,
        noise_map=dataset.positions_noise_map,
        tracer=tracer,
        solver=None,
        profile=point_profile,
    )
    sum_library_log_likelihood_source += float(fit.log_likelihood)

print(f"Library source-plane log likelihood = {sum_library_log_likelihood_source:.4e}")
print(
    f"Match: {np.isclose(log_likelihood_source, sum_library_log_likelihood_source, rtol=1e-6)}"
)


"""
__Image-Plane Chi Squared: Concept__

The image-plane chi² takes the opposite approach. Instead of ray-tracing observed positions to
the source plane and measuring source-plane scatter, it **forward-solves** the model's image-plane
positions for each source and compares them to the observed positions in the image plane.

Mechanically:

 1. Take the source-plane centre (model's ``Point.centre`` per source).
 2. Use a ``PointSolver`` to ray-trace triangles from a fine image-plane grid forward through the
    lens until they converge to that source-plane centre — these are the **model positions**.
 3. For each model position, find the closest observed position and pair them.
 4. The residual per pair is just the image-plane Euclidean distance.

The chi² is then ``residual² / noise²`` — no magnification weighting, because the residual is
already in the same units (arc-seconds) as the noise. This makes the absolute chi² much smaller
than the source-plane variant at the same model: at the truth, residuals are bounded by the
``PointSolver`` precision (~0.001"), so chi² ≈ (0.001 / 0.005)² × N ≈ N × 0.04, where N is the
number of paired images. Far below the source-plane chi² which is dominated by magnification × the
same precision floor.

The trade-off: image-plane chi² requires a **forward solve per evaluation**, which is slow,
introduces solver precision as a noise floor, and carries pairing pathologies discussed below.
The source-plane chi² has none of those costs.

__Point Solver Setup__

The ``PointSolver`` searches for image-plane positions whose forward ray-trace lands at the
source-plane target. Conceptually it does this by:

 1. Starting from a coarse image-plane grid.
 2. Tessellating into triangles. For each triangle, ray-trace its three vertices to the source
    plane. If the source-plane target lies inside the traced triangle, that image-plane triangle
    contains a model position.
 3. Refining each containing triangle — subdivide, ray-trace the sub-triangles, repeat — until
    the triangle size drops below a configurable ``pixel_scale_precision``.
 4. The triangle centroid at convergence is the model position.

This is iterative and the runtime scales with the source-plane precision target. The solver also
filters out central images via ``magnification_threshold``: highly demagnified images (often the
unobservable central image of a strong-lens configuration) are discarded so the model doesn't pair
unphysical demagnified solutions to the observed multi-image set.

A standalone walkthrough of the triangle-refinement algorithm (sub-grid traversal, magnification
filtering, multi-plane handling, JAX-compatibility) has not yet been written. The closest reference
is ``autolens_workspace/scripts/guides/point_source_pairing.py``, which documents the pairing
schemes and solver settings the refinement feeds.

In code, the solver is constructed once and reused per evaluation:
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=1.0),
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)


"""
__Forward Solving Model Positions__

For each source, call ``solver.solve(tracer, source_plane_coordinate=...)`` with the source's
truth source-plane centre. The solver returns the image-plane (y, x) positions of every
multiple image whose magnification is above ``magnification_threshold``.

Multi-plane ray tracing happens *inside* ``solve()`` automatically. For source_1 (z=2.0) the
triangles are forward-traced through the z=1.0 source_0 plane before reaching z=2.0 — the same
recursive lens equation discussed above is applied during the forward solve. The solver picks the
right target plane via the ``plane_redshift`` argument: this is the source galaxy's redshift, and
the solver maps it to the corresponding plane index in the tracer.
"""
model_positions_per_source = []
for i, dataset in enumerate(dataset_list):
    point_profile = getattr(source_galaxies[i], dataset.name)
    model_positions = solver.solve(
        tracer=tracer,
        source_plane_coordinate=point_profile.centre,
        plane_redshift=dataset.redshift,
    )
    model_positions_per_source.append(model_positions)
    print(f"  {dataset.name}: model positions = {model_positions.in_list}")


"""
__Pairing Model to Observed__

We have two sets of image-plane positions per source: the **model positions** (forward-solved) and
the **observed positions** (from ``point_datasets.csv``). The chi² requires a one-to-one mapping
between them — but the two sets don't come pre-paired, and may not even have the same length.

**Three pairing schemes** exist in PyAutoLens, each with different behaviour when the counts
don't match:

 1. **``FitPositionsImagePair`` (Hungarian, no-repeat).** Pairs each observed to its nearest
    predicted image via the **Hungarian algorithm** (also called linear sum assignment). The
    algorithm finds the unique 1-to-1 pairing that *globally minimises the total distance* —
    not greedy. Two consequences: (a) it gives the optimal assignment even when greedy would
    fail; (b) when counts differ, unmatched positions still get unpaired and contribute nothing
    to chi².

 2. **``FitPositionsImagePairAll`` (closest, with replacement).** Each model position pairs to its
    nearest observed position independently. A single observed position may be paired to multiple
    model positions. Every model position contributes to chi².

 3. **``FitPositionsImagePairRepeat`` (repeats allowed).** Similar to (2) but with explicit
    handling of cases where the model genuinely predicts multiple images at nearly the same
    location (e.g. near a caustic crossing).

This walkthrough uses scheme (1) — ``FitPositionsImagePair`` — which is the historical default and
the one ``al.FitPointDataset`` constructs by default. We follow that convention.

**The known pathology** (documented in the PyAutoLens source for ``FitPositionsImagePair``):

 - **Model predicts too many images** (more model than observed). The pairing leaves some model
   positions unpaired. These model positions contribute *nothing* to chi². A model that
   spuriously generates extra demagnified images can therefore have its chi² artificially
   *reduced* compared to a model that produces exactly the observed count.
 - **Model predicts too few images** (fewer model than observed). Some observed positions go
   unpaired and also contribute nothing. The chi² is similarly artificially reduced.

In both cases the optimiser can prefer pathological solutions with the wrong image count over the
correct solution. ``magnification_threshold`` partially mitigates this by discarding very
demagnified spurious images, but it is not a complete fix. ``FitPositionsImagePairAll`` /
``...Repeat`` address some of the cases, at the cost of other failure modes. Selecting the right
scheme for a real cluster fit is a per-dataset judgement call.

In code we do the same pairing the library does — Hungarian / linear sum assignment — by
computing the pairwise distance matrix and handing it to ``scipy.optimize.linear_sum_assignment``:
"""
from scipy.optimize import linear_sum_assignment


def _pair_hungarian(model_positions, observed_positions):
    """Optimal 1-1 pairing via Hungarian algorithm.

    Returns a list of (model_position, observed_position, distance) tuples. When the number of
    model and observed positions differ, only ``min(N_model, N_observed)`` pairs are returned
    — the excess positions on the larger side go unpaired and contribute nothing to chi².
    """
    model_arr = np.asarray(model_positions)
    observed_arr = np.asarray(observed_positions)

    # Pairwise distance matrix: distances[i, j] = ||model_i - observed_j||.
    distances = np.linalg.norm(
        model_arr[:, np.newaxis, :] - observed_arr[np.newaxis, :, :], axis=2
    )

    row_ind, col_ind = linear_sum_assignment(distances)

    return [
        (model_arr[i], observed_arr[j], float(distances[i, j]))
        for i, j in zip(row_ind, col_ind)
    ]


pairs_per_source = []
for i, dataset in enumerate(dataset_list):
    pairs = _pair_hungarian(model_positions_per_source[i], dataset.positions)
    pairs_per_source.append(pairs)
    print(
        f"  {dataset.name}: {len(pairs)} pairs / {len(dataset.positions)} observed / "
        f"{len(model_positions_per_source[i])} model"
    )


"""
__Image-Plane Residual Map__

Residual per pair = image-plane Euclidean distance between model and observed (in arc-seconds).
This is what ``_pair_hungarian`` already returned as the third element of each tuple.
"""
residual_maps_image_per_source = []
for i, pairs in enumerate(pairs_per_source):
    residuals = np.array([p[2] for p in pairs])
    residual_maps_image_per_source.append(residuals)


"""
__Chi Squared Map (Image)__

Image-plane chi² per pair:

.. math::

    \\chi^{2}_{i} = \\frac{r_{i}^{2}}{\\sigma_{i}^{2}}

No magnification weighting — the residual is already in image-plane arc-seconds, the same units
as the noise.

Each pair should use the noise of its paired observed position. For simplicity we use the dataset's
mean noise, since all positions in the cluster simulator share the same σ; a real analysis should
index per-observed-position.
"""
chi_squared_maps_image_per_source = []
for i, dataset in enumerate(dataset_list):
    r = residual_maps_image_per_source[i]
    sigma = float(np.mean(np.asarray(dataset.positions_noise_map)))
    chi_sq_map = r**2 / sigma**2
    chi_squared_maps_image_per_source.append(chi_sq_map)


"""
__Per-Source / Total Chi Squared (Image)__
"""
chi_squared_per_source_image = [
    float(np.sum(m)) for m in chi_squared_maps_image_per_source
]
total_chi_squared_image = float(sum(chi_squared_per_source_image))
print(f"Total image-plane chi² = {total_chi_squared_image:.4e}")


"""
__Noise Normalization (Image)__

The Gaussian-log normalisation for image-plane chi² is the standard form:

.. math::

    \\mathcal{N} = \\sum_{i} \\log \\left( 2\\pi \\, \\sigma_{i}^{2} \\right)

No magnification factor (residuals are already in image-plane units).
"""
noise_normalizations_per_source_image = []
for i, dataset in enumerate(dataset_list):
    n_pairs = len(residual_maps_image_per_source[i])
    sigma = float(np.mean(np.asarray(dataset.positions_noise_map)))
    nn = float(n_pairs * np.log(2 * np.pi * sigma**2))
    noise_normalizations_per_source_image.append(nn)

total_noise_normalization_image = float(sum(noise_normalizations_per_source_image))
print(f"Total image-plane noise normalization = {total_noise_normalization_image:.4e}")


"""
__Image-Plane Log Likelihood__
"""
log_likelihood_image = -0.5 * (
    total_chi_squared_image + total_noise_normalization_image
)
print(f"Image-plane log likelihood = {log_likelihood_image:.4e}")


"""
__Image-Plane Validation__

Instantiate ``al.FitPositionsImagePair`` and confirm match. Note: ``FitPositionsImagePair`` uses
the same Hungarian (linear-sum-assignment) pairing scheme as our manual implementation above, so the
chi² values should agree to within numerical precision (the library's solver may use slightly
different triangle precision at refinement edge cases, producing sub-percent residual differences).
"""
sum_library_log_likelihood_image = 0.0
for i, dataset in enumerate(dataset_list):
    point_profile = getattr(source_galaxies[i], dataset.name)
    fit = al.FitPositionsImagePair(
        name=dataset.name,
        data=dataset.positions,
        noise_map=dataset.positions_noise_map,
        tracer=tracer,
        solver=solver,
        profile=point_profile,
    )
    sum_library_log_likelihood_image += float(fit.log_likelihood)

print(f"Library image-plane log likelihood = {sum_library_log_likelihood_image:.4e}")
print(
    f"Match (rtol=1e-2): "
    f"{np.isclose(log_likelihood_image, sum_library_log_likelihood_image, rtol=1e-2)}"
)


"""
__Source-Plane vs Image-Plane: When to Use Which__

| Aspect | Source-plane (``FitPositionsSource``) | Image-plane (``FitPositionsImagePair``) |
|---|---|---|
| **Forward solve required?** | No | Yes (one per evaluation) |
| **Per-evaluation cost** | Cheap (just back-projection) | Expensive (triangle refinement) |
| **JAX-friendly?** | Yes | The solver is JAX-jitted, but each evaluation triggers a forward solve regardless |
| **Magnification weighting** | Yes (``μ²`` in chi²) | No |
| **Sensitive to ``PointSolver`` precision?** | Amplified by ``μ²`` at multi-image positions | Bounded by precision directly |
| **Pairing pathology** | None (uses all images) | Yes (too-many / too-few; see above) |
| **Absolute chi² scale (at truth)** | Large (~``μ²`` × precision²) | Small (~precision²) |
| **Best for** | Fast Nautilus fits; JAX-jit'd parameter estimation | Final residual visualisation; cases where pairing is unambiguous |

For most cluster fits the source-plane chi² is the right default: it's faster, JAX-compatible,
and doesn't suffer pairing pathologies. Its solved sibling ``FitPositionsSourceSolved`` (paired
with ``al.ps.PointSolved`` sources) sharpens this further — the source centres drop out of the
non-linear space entirely (2 parameters per source) at timing-noise-level extra cost per call,
and its tensor weighting is a better error model than the scalar ``μ²`` used above. The
recommended cluster workflow is therefore: **search with ``FitPositionsSourceSolved``, validate
with the image-plane chi²** on the max-likelihood model. The image-plane chi² remains the
diagnostic tool — visualising where predicted images sit relative to observed ones, and
cross-checking any source-plane bias.

This is exactly what the cluster modeling scripts (``scripts/cluster/start_here.py`` and
``scripts/cluster/modeling.py``) demonstrate: their ``AnalysisPoint`` analyses pass
``fit_positions_cls=al.FitPositionsSourceSolved`` with parameter-free ``al.ps.PointSolved``
sources (the class default remains ``FitPositionsImagePairRepeat``).

__Wrap Up__

This script has walked through the cluster point-source likelihood end to end, in both source-plane
and image-plane flavours, for a multi-plane (2-source) cluster.

Next steps:

- ``scripts/cluster/csv_api.py`` — the family CSV schema this script reads.
- ``scripts/cluster/simulator.py`` — how the truth dataset was generated.
- ``scripts/cluster/modeling.py`` — full Nautilus fit using the same likelihood function discussed
  here, with priors and ``AnalysisPoint`` configuration.
- ``autolens_workspace_test/scripts/cluster/likelihood_sanity.py`` — perturbation sweep that
  surfaces the ``PointSolver`` precision-floor pathology described in the source-plane section.
- ``autolens_workspace/scripts/guides/advanced/multi_plane.py`` — full derivation of the recursive
  lens equation.
- ``autolens_workspace/scripts/guides/lens_calc.py`` — Hessian / magnification /
  critical-curve mechanics.

For a deeper understanding of cluster lens modelling and point-source likelihoods, the
**HowToLens** Jupyter notebook lectures cover both topics in detail.
"""
