"""
Potential Corrections (Gravitational Imaging)
=============================================

Parametric lens mass models (e.g. the `PowerLaw` used throughout the workspace) are smooth: their convergence and
deflection angles vary gradually across the image plane. Real lens galaxies are not perfectly smooth. They contain
substructure — dark matter subhaloes, globular clusters, satellite galaxies — and their large-scale mass
distributions can depart from ellipsoidal symmetry in ways no simple parametric profile captures.

Gravitational imaging, also called the potential correction method, reconstructs these departures directly. Small
pixelized corrections $\delta\psi$ to the lensing potential are defined on a coarse regular mesh over the image
plane, and solved for linearly alongside the pixelized source, by maximising the Bayesian evidence. Wherever the
smooth model fails to fit the lensed arcs, the corrections absorb the discrepancy — and their curvature
$\delta\kappa = \frac{1}{2}\nabla^2 \delta\psi$ maps the missing (or excess) convergence, revealing substructure
without assuming its form.

This technique detected a $\sim 10^6 \, M_\odot$ object at cosmological distance in the VLBI imaging of
JVAS B1938+666 (Powell et al. 2025, Nature Astronomy 9, 1714; Vegetti et al. 2026, Nature Astronomy), the smallest
dark object ever found by lensing. The implementation in **PyAutoLens** (`al.pc`) is ported from the
`potential_correction` package of Cao et al. 2025 (https://github.com/caoxiaoyue/lensing_potential_correction).
If you use it in your research, please cite Cao et al. 2025 — citation materials are at
https://github.com/caoxiaoyue/potential_correction_paper.

__Overview__

The method models the observed image $d$ as the smooth-model image plus a linear response to the corrections:

$\delta d = - B \, D_s \, D_\psi \, \delta\psi$

where $B$ is the PSF blur matrix, $D_s$ holds the source's brightness gradients at the ray-traced position of every
image pixel, and $D_\psi$ interpolates the coarse $\delta\psi$ mesh onto the image grid and takes its spatial
gradients. Intuitively: a small change to the potential deflects a ray slightly, which samples the source at a
slightly different position, which changes the observed brightness in proportion to the source's local gradient.

Both the source and $\delta\psi$ are regularized, and the strengths of both regularizations are set objectively by
maximising the Bayesian evidence — no manual tuning, following Cao et al. 2025 (building on Koopmans 2005;
Vegetti & Koopmans 2009; Suyu et al. 2009; Vernardos & Koopmans 2022).

__Contents__

- **Simulate:** Simulate strong lens imaging whose mass distribution contains a dark subhalo the smooth model omits.
- **Smooth Model Fit:** Fit the data with the smooth (subhalo-free) mass model, whose residuals localize the subhalo.
- **Joint Fit:** Reconstruct the source and potential corrections jointly with `FitDpsiSrcImaging`.
- **Dkappa Map:** Convert the corrections to a convergence map and compare to the true subhalo.
- **Iterative Fit:** Refine the corrections with the iterative Levenberg-Marquardt engine `IterFitDpsiSrcImaging`.
- **Evidence Sampling:** Sample the regularization hyper-parameters with a non-linear search via `DpsiSrcInvAnalysis`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Simulate__

We simulate imaging of a strong lens whose mass model is an `Isothermal` plus a $10^{10} \, M_\odot$ NFW dark
subhalo sitting right on the Einstein ring, where its imprint on the lensed arcs is strongest.

The source is a compact double-knot galaxy, giving the arcs the sharp brightness gradients that gravitational
imaging leverages (the response to $\delta\psi$ scales with the source gradient).
"""
grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.05, over_sample_size=4)

psf = al.Convolver.from_gaussian(shape_native=(11, 11), sigma=0.05, pixel_scales=0.05)

simulator = al.SimulatorImaging(
    exposure_time=840.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.4,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=0.0),
    ),
    subhalo=al.mp.NFWMCRLudlowSph(
        centre=(1.41, 0.0),
        mass_at_200=1.0e10,
        redshift_object=0.2,
        redshift_source=0.6,
    ),
)

source_true = al.Galaxy(
    redshift=0.6,
    bulge0=al.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=45.0),
        intensity=5.0,
        sigma=0.15,
    ),
    bulge1=al.lp.Gaussian(
        centre=(0.0, 0.4),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.4, angle=135.0),
        intensity=3.0,
        sigma=0.1,
    ),
)

tracer_true = al.Tracer(galaxies=[lens_true, source_true])
dataset = simulator.via_tracer_from(tracer=tracer_true, grid=grid)

"""
__Arc Mask__

Gravitational imaging only constrains the potential where the lensed arcs are: the corrections respond to the data
through the source gradients, which vanish off the arcs. We therefore mask the data to an arc-tracing region using
`al.pc.util.arc_mask_from`, which thresholds the signal-to-noise map, drops small disconnected islands, dilates the
result and cleans it so every unmasked pixel supports the finite-difference derivative operators.
"""
mask_array = al.pc.util.arc_mask_from(
    np.asarray(dataset.signal_to_noise_map.native),
    threshold=3.0,
    ignore_size=25,
    ext_size=5,
)
mask = al.Mask2D(mask=mask_array, pixel_scales=dataset.pixel_scales)
masked_imaging = dataset.apply_mask(mask=mask)

"""
__Smooth Model Fit__

The starting point of a potential-correction analysis is the best smooth-model fit — in a real analysis, the result
of a standard lens-modeling pipeline. Its residuals concentrate near the subhalo, because the smooth model cannot
bend the arcs the way the subhalo does.

Here we cheat and use the true smooth mass model (the `Isothermal` without the subhalo) and the true source, which
isolates the technique itself; a real analysis would use the maximum-likelihood parametric model.

The source enters the correction operator through an `al.pc.SrcFactory`, which evaluates the source's brightness and
gradients at arbitrary source-plane positions. For a parametric source we use `AnalyticSrcFactory`; a pixelized
reconstruction from a previous fit would use `PixSrcFactoryITP`.
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

"""
__Joint Fit__

`FitDpsiSrcImaging` inverts the image jointly for the pixelized source and the potential corrections, fully
accounting for their covariance. Its components are:

- `DpsiPixelization`: the corrections' mesh (`RegularDpsiMesh(factor=2)` — a mesh twice as coarse as the data grid)
  and their regularization. `aa.reg.MaternKernel` (with `nu=2.5`) flexibly permits both localised and extended
  perturbations (the key advance of Cao et al. 2025 over curvature-only schemes); `aa.reg.CurvatureMask` and
  `aa.reg.FourthOrderMask` are the classic alternatives.

- The source pixelization: any standard PyAutoLens pixelization; here a k-nearest-neighbour mesh distributed by an
  `Overlay` image mesh.
"""
dpsi_pixelization = al.pc.DpsiPixelization(
    mesh=al.pc.RegularDpsiMesh(factor=2),
    regularization=al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5),
)

grid_slim = masked_imaging.grid.slim
source_shape = (
    int(float(grid_slim[:, 0].max() - grid_slim[:, 0].min()) / 0.05 / 2.0),
    int(float(grid_slim[:, 1].max() - grid_slim[:, 1].min()) / 0.05 / 2.0),
)
src_pixelization = al.Pixelization(
    mesh=al.mesh.KNearestNeighbor(pixels=int(np.prod(source_shape))),
    regularization=al.reg.Constant(coefficient=3.8),
)
src_image_mesh = al.image_mesh.Overlay(shape=source_shape)

fit = al.pc.FitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
)

print(f"joint source + dpsi log evidence = {fit.log_evidence:.4e}")

"""
__Dkappa Map__

The reconstructed corrections live on the coarse dpsi mesh (`fit.best_fit_dpsi`). Their physical meaning is clearest
as a convergence correction, obtained by applying the mesh's Laplacian operator:

$\delta\kappa = \frac{1}{2} \nabla^2 \delta\psi$

A dark subhalo the smooth model omits appears as a positive $\delta\kappa$ peak at its position — compare the map
below to the true subhalo at (y, x) = (1.41", 0.0").

The nine-panel summary figure shows the data, model, residuals, the $\delta\psi$ and $\delta\kappa$ maps and the
source reconstruction.
"""
dkappa = fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.best_fit_dpsi

peak = np.argmax(dkappa)
print(
    f"dkappa peak at (y, x) = ({fit.pair_dpsi_data_obj.ygrid_dpsi_1d[peak]:.2f}, "
    f"{fit.pair_dpsi_data_obj.xgrid_dpsi_1d[peak]:.2f}) — true subhalo at (1.41, 0.00)"
)

al.pc.visualize.show_fit_dpsi_src(
    fit, output=path.join("output", "potential_correction_joint_fit.png")
)

"""
__Iterative Fit__

The joint fit above linearizes the corrections around the smooth model once. The iterative engine
`IterFitDpsiSrcImaging` goes further, following the approach of Powell et al. 2025: it optimizes the combined
state [source | dpsi] with a Levenberg-Marquardt loop, and after every accepted step **re-ray-traces the image grid
through the corrected lens** (the smooth model plus an `InputPotential` mass profile built from the current
corrections). The corrections thus feed back into the source mapping, capturing compact perturbers more faithfully
than a single linearization.

`gauge_constraints=True` removes the degeneracies inherent to potential corrections — a constant $\delta\psi$ and
linear gradients (which just shift the source) are unconstrained by the data — by enforcing
$\langle\delta\psi, 1\rangle = \langle\delta\psi, x\rangle = \langle\delta\psi, y\rangle = 0$ at every step.

The dense linear algebra runs through `al.pc.dense_util`, whose kernels follow the PyAuto `xp` convention: pass
`xp=jax.numpy` to `solve_joint_optimization` to run them under JAX on an accelerator; the default is numpy.
"""
iter_fit = al.pc.IterFitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
    gauge_constraints=True,
    n_iter=5,
)

s_opt, dpsi_opt = iter_fit.solve_joint_optimization()
print(f"iterative Laplace log evidence = {iter_fit.log_evidence():.4e}")

dkappa_iter = iter_fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_opt
peak = np.argmax(dkappa_iter)
print(
    f"iterative dkappa peak at (y, x) = ({iter_fit.pair_dpsi_data_obj.ygrid_dpsi_1d[peak]:.2f}, "
    f"{iter_fit.pair_dpsi_data_obj.xgrid_dpsi_1d[peak]:.2f})"
)

"""
__Evidence Sampling__

In a real analysis the regularization hyper-parameters (the coefficients and scales above) are not known. They are
sampled with a non-linear search, using the inversion's Bayesian evidence as the likelihood, via the
`DpsiSrcInvAnalysis` (one-shot) or `IterDpsiSrcInvAnalysis` (iterative) analysis classes:

    dpsi_model = af.Model(
        al.pc.DpsiSrcPixelization,
        dpsi_pixelization=af.Model(
            al.pc.DpsiPixelization,
            mesh=al.pc.RegularDpsiMesh(factor=2),
            regularization=af.Model(al.reg.MaternKernel, nu=2.5),
        ),
        src_pixelization=src_pixelization,
    )

    analysis = al.pc.DpsiSrcInvAnalysis(
        masked_imaging=masked_imaging,
        lens_start=lens_smooth,
        source_start=source_start,
        src_image_mesh=src_image_mesh,
    )

    search = af.Nautilus(name="potential_correction", n_live=100)
    result = search.fit(model=dpsi_model, analysis=analysis)

This recovers both localised subhaloes and extended perturbations (e.g. Gaussian-random-field departures — see
`ag.mp.GaussianRandomField` for simulating them) with the regularization set objectively by the data.

__Interferometer__

Gravitational imaging's benchmark detections are radio/VLBI (the B1938+666 results above), and `al.pc` supports
`Interferometer` datasets in visibility space through the **sparse-operator (w-tilde) route**, whose cost scales
with real-space mask pixels independent of the visibility count. The certified recipe (validated on realistic
earth-rotation-synthesis coverage at ~10^4 visibilities, where both engines recover a simulated
$10^{10} M_\odot$ subhalo at ~9$\sigma$ with the dkappa peak 0.34" from the truth) is:

    # the real-space mask stays a filled circle (it defines the FFT extent);
    # the corrections are restricted to an arc-tracing sub-mask
    dataset = dataset.apply_sparse_operator()   # precompute the w-tilde operator (cache it to disk!)

    fit = al.pc.FitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens_smooth,
        source_start=source_start,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        dpsi_mask=arc_dpsi_mask,
    )
    evidence = fit.log_evidence            # one-shot joint inversion (sparse route)

    iter_fit = al.pc.IterFitDpsiSrcInterferometer(
        dataset=dataset,
        lens_start=lens_smooth,
        dpsi_pixelization=dpsi_pixelization,
        src_pixelization=src_pixelization,
        dpsi_mask=arc_dpsi_mask,
        gauge_constraints=True,
        reg_optimize_every=1,              # evidence-control the regularizations each step
    )
    s_opt, dpsi_opt = iter_fit.solve_joint_optimization(
        x0=fit.src_dpsi_slim,              # warm-start from the one-shot solution
    )

Two practices matter, both mirroring the published pipelines: **warm-start the iterative engine from the
one-shot solution** (it then refines inside the right basin instead of searching from zero), and let the
**evidence control the regularization strengths** (`reg_optimize_every`) rather than fixing them — the Bayesian
evidence consistently ranks over-fit or degenerate solutions below the recovering ones, so evidence-driven
selection self-protects. Always check first that your smooth-model source fit reaches chi-squared per degree of
freedom near ~1-3: outside that regime the corrections absorb source-model error rather than real structure.

__Wrap Up__

Gravitational imaging turns the residuals of a smooth lens model into a map of the missing mass. The `al.pc`
subpackage provides the one-shot joint inversion, the iterative LM engine, and the analysis classes to sample their
hyper-parameters — with all linear algebra available under numpy or JAX through the `xp` API.

If you use this functionality, please cite Cao et al. 2025
(https://github.com/caoxiaoyue/potential_correction_paper) alongside PyAutoLens.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
