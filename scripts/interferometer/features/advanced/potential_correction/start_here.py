"""
Feature: Potential Correction (Gravitational Imaging)
=====================================================

This example performs gravitational imaging on interferometer data: pixelized corrections $\delta\psi$ to the
lensing potential are reconstructed jointly with the pixelized source, revealing mass structure (e.g. dark
subhaloes) the smooth lens model omits, as a convergence-correction map $\delta\kappa = \frac{1}{2}\nabla^2\delta\psi$.

The visibility-space implementation uses the **sparse-operator (w-tilde) route**, whose cost scales with the
number of real-space mask pixels — independent of the number of visibilities — making it the path for real
datasets (ALMA / VLBI scale). This is the regime of the technique's benchmark detections: the ~$10^6 M_\odot$
object in JVAS B1938+666 (Powell et al. 2025, Nat. Astron. 9, 1714; Vegetti et al. 2026).

The workflow below is the configuration certified by the PyAutoLens validation campaign, which on this exact
setup recovers the simulated $10^{10} M_\odot$ subhalo with dkappa correlation ~0.83 against the truth, the
peak ~0.15" from the true position and ~6 sigma significance over the field:

 1. verify the smooth-model source fit reaches $\chi^2$/dof ~ 1-3 (the "regime gate" — corrections only
    measure residual mass structure once the source model fits the data to near the noise);
 2. run the one-shot joint inversion (`FitDpsiSrcInterferometer`) with an arc-restricted `dpsi_mask`;
 3. refine with the iterative engine **warm-started from the one-shot**, with the regularization strengths
    re-optimized by Bayesian evidence at every step (the Koopmans 2005 / Vegetti & Koopmans 2009 scheme).

If you use this functionality in your research, please cite Cao et al. 2025, whose `potential_correction`
package (https://github.com/caoxiaoyue/lensing_potential_correction) the implementation is ported from;
citation materials are provided at https://github.com/caoxiaoyue/potential_correction_paper.

This script takes a few minutes to run (the one-off sparse-operator precomputation dominates).

__Prerequisites__

- `imaging/features/advanced/potential_correction/start_here.ipynb` — the technique overview (imaging).
- `interferometer/start_here.ipynb` — interferometer dataset basics.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import numpy as np

import autoarray as aa
import autolens as al

"""
__UV Coverage__

We simulate a small but realistic dataset in-memory (seeded, so this script is fully reproducible). Random uv
points produce heavy sidelobes that smear the corrections, so we build an ALMA-like earth-rotation-synthesis
distribution: Gaussian-distributed antennas observed over an hour-angle track, giving the dense elliptical-ring
uv sampling of a real observation. The maximum baseline is kept at ~60% of the real-space grid's Nyquist limit.

For real data, load your visibilities via `al.Interferometer.from_fits` as in `interferometer/start_here.py`.
"""


def synthesis_uv_from(n_ant, n_times, max_baseline_wavelengths, seed=0):
    rng = np.random.default_rng(seed)
    antennas = rng.normal(scale=max_baseline_wavelengths / 4.0, size=(n_ant, 2))
    i_idx, j_idx = np.triu_indices(n_ant, k=1)
    baselines = antennas[i_idx] - antennas[j_idx]

    hour_angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, n_times)
    declination = np.deg2rad(-23.0)

    uv_list = []
    for ha in hour_angles:
        u = baselines[:, 0] * np.cos(ha) - baselines[:, 1] * np.sin(ha)
        v = baselines[:, 0] * np.sin(ha) * np.sin(declination) + baselines[
            :, 1
        ] * np.cos(ha) * np.sin(declination)
        uv_list.append(np.stack([u, v], axis=1))
    return np.concatenate(uv_list, axis=0)


pixel_scale = 0.08
nyquist = 1.0 / (2.0 * pixel_scale * np.pi / 180.0 / 3600.0)
uv_wavelengths = synthesis_uv_from(
    n_ant=18, n_times=8, max_baseline_wavelengths=0.6 * nyquist
)
print(f"{uv_wavelengths.shape[0]} visibilities")

"""
__Simulate__

An `Isothermal` lens whose true mass also contains a $10^{10} M_\odot$ NFW subhalo on the Einstein ring, and a
compact double-Gaussian source.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(72, 72), pixel_scales=pixel_scale, radius=2.6
)
grid = al.Grid2D.from_mask(mask=real_space_mask)

subhalo_centre = (1.41, 0.0)
true_subhalo = al.mp.NFWMCRLudlowSph(
    centre=subhalo_centre, mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
)
lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.4,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=0.0),
    ),
    subhalo=true_subhalo,
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

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=2.0,
    noise_seed=1,
)
dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)
dataset = al.Interferometer(
    data=dataset.data,
    noise_map=dataset.noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
)

"""
__Sparse Operator__

The w-tilde machinery is precomputed once per dataset. For large visibility counts this is the expensive step
(it is also noise-map dependent) — for repeated analyses of the same dataset, cache it to disk and reload via
`apply_sparse_operator(nufft_precision_operator=np.load(...))`.
"""
dataset = dataset.apply_sparse_operator()

"""
__Smooth Starting Model__

Potential corrections perturb a smooth starting model — in a real analysis, your maximum-likelihood parametric
fit (see `interferometer/modeling`). Here we use the true smooth lens (without the subhalo) and the true
source, isolating the technique itself. The source enters the correction operator through a `SrcFactory`: an
`AnalyticSrcFactory` here; use `PixSrcFactoryITP` to build it from a pixelized reconstruction of a previous fit.
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

"""
__Source Pixelization + Regime Gate__

The source is reconstructed on a k-nearest-neighbor mesh distributed via an `Overlay` image-mesh. Before
trusting any correction, verify the smooth model + this source pixelization actually fit the data: if
$\chi^2$/dof is far above ~1-3, the corrections will absorb source-model error instead of mass structure and
the dkappa map is meaningless (this gate is the single most common failure mode in practice).
"""
grid_slim = dataset.grid.slim
source_shape = (
    int(float(grid_slim[:, 0].max() - grid_slim[:, 0].min()) / pixel_scale / 2.0),
    int(float(grid_slim[:, 1].max() - grid_slim[:, 1].min()) / pixel_scale / 2.0),
)
src_pixelization = al.Pixelization(
    mesh=al.mesh.KNearestNeighbor(pixels=int(np.prod(source_shape))),
    regularization=al.reg.Constant(coefficient=1.0),
)
src_image_mesh = al.image_mesh.Overlay(shape=source_shape)

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages

source_galaxy_pix = al.Galaxy(redshift=0.6, pixelization=src_pixelization)
image_plane_mesh_grid = src_image_mesh.image_plane_mesh_grid_from(mask=real_space_mask)
smooth_fit = al.FitInterferometer(
    dataset=dataset,
    tracer=al.Tracer(galaxies=[lens_smooth, source_galaxy_pix]),
    adapt_images=AdaptImages(
        galaxy_image_plane_mesh_grid_dict={source_galaxy_pix: image_plane_mesh_grid}
    ),
    settings=aa.Settings(use_positive_only_solver=True, use_border_relocator=True),
)
chi2_dof = float(smooth_fit.chi_squared) / (2 * uv_wavelengths.shape[0])
print(f"regime gate: smooth-model chi2/dof = {chi2_dof:.2f} (target ~1-3)")

"""
__Arc-Restricted Dpsi Mesh__

The real-space mask must stay a filled circle (it defines the sparse operator's FFT extent), but the
corrections are only constrained where the lensed arcs are — restrict the dpsi mesh to an arc-tracing sub-mask
built from the smooth-model image geometry.
"""
tracer_smooth = al.Tracer(galaxies=[lens_smooth, source_true])
arc_image = np.asarray(tracer_smooth.image_2d_from(grid=grid).native)
arc_mask = al.pc.util.arc_mask_from(
    arc_image / (0.05 * arc_image.max()), threshold=3.0, ignore_size=10, ext_size=3
)
dpsi_mask = ~((~arc_mask) & (~np.asarray(real_space_mask)))
print(f"dpsi mesh restricted to {int(np.count_nonzero(~dpsi_mask))} arc pixels")

"""
__One-Shot Joint Inversion__

The pixelized source and dpsi are solved in one linear system through the sparse route, with Matern
regularization on the corrections (the Cao et al. 2025 scheme, which recovers localised perturbers where
curvature penalties smear them out).
"""
dpsi_pixelization = al.pc.DpsiPixelization(
    mesh=al.pc.RegularDpsiMesh(factor=2),
    regularization=al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5),
)

fit = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
    dpsi_mask=dpsi_mask,
    use_sparse_operator=True,
)
print(f"one-shot joint log evidence = {fit.log_evidence:.4e}")

"""
__Dkappa Recovery Metrics__

Because this is a simulation we can compare the recovered convergence correction to the true subhalo's
convergence: the correlation over the dpsi mesh, the peak's offset from the true position, and the peak's
significance over the field far from the subhalo.
"""


def dkappa_metrics(pair_obj, dkappa_rec, tag):
    points = np.vstack([pair_obj.ygrid_dpsi_1d, pair_obj.xgrid_dpsi_1d]).T
    dkappa_true = np.asarray(
        true_subhalo.convergence_2d_from(grid=al.Grid2DIrregular(values=points))
    )
    corr = float(np.corrcoef(dkappa_rec, dkappa_true)[0, 1])
    peak = points[int(np.argmax(dkappa_rec))]
    dist = float(np.hypot(peak[0] - subhalo_centre[0], peak[1] - subhalo_centre[1]))
    r = np.hypot(points[:, 0] - subhalo_centre[0], points[:, 1] - subhalo_centre[1])
    significance = float(dkappa_rec.max() / dkappa_rec[r > 1.5].std())
    print(
        f"{tag}: corr(dkappa_rec, dkappa_true) = {corr:.3f}; "
        f'peak offset = {dist:.2f}"; significance = {significance:.1f} sigma'
    )
    return corr, dist


dkappa = np.asarray(fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.best_fit_dpsi)
dkappa_metrics(fit.pair_dpsi_data_obj, dkappa, "one-shot")

"""
__Iterative Refinement__

The iterative Levenberg-Marquardt engine re-ray-traces through the corrected lens at each accepted step. Two
practices are essential (both certified by the validation campaign, mirroring the published pipelines):

- **warm-start from the one-shot solution** (`x0=`), so the LM refines inside the right basin;
- **evidence-control the regularization strengths** (`reg_optimize_every=1`) rather than fixing them.

Under JAX, pass `xp=jax.numpy` (with 64-bit enabled) to run the LM kernels JIT-compiled on GPU — the API is
identical, per the ecosystem's xp convention.
"""
iter_fit = al.pc.IterFitDpsiSrcInterferometer(
    dataset=dataset,
    lens_start=lens_smooth,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
    dpsi_mask=dpsi_mask,
    gauge_constraints=True,
    n_iter=6,
    reg_optimize_every=1,
)
s_opt, dpsi_opt = iter_fit.solve_joint_optimization(x0=np.asarray(fit.src_dpsi_slim))
print(
    f"iterative Laplace log evidence = {iter_fit.log_evidence(s=s_opt, dpsi=dpsi_opt):.4e}"
)

dkappa_iter = np.asarray(iter_fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_opt)
dkappa_metrics(iter_fit.pair_dpsi_data_obj, dkappa_iter, "iterative")

"""
__Dkappa Map__

The convergence-correction map — the subhalo appears as a positive peak at its position (1.41", 0.0").
"""
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
al.pc.visualize.imshow_masked_data(
    dkappa,
    fit.pair_dpsi_data_obj.mask_dpsi,
    ax=axes[0],
    origin="upper",
    extent=fit.pair_dpsi_data_obj.data_bound,
)
axes[0].set_title("dkappa (one-shot)")
al.pc.visualize.imshow_masked_data(
    dkappa_iter,
    iter_fit.pair_dpsi_data_obj.mask_dpsi,
    ax=axes[1],
    origin="upper",
    extent=iter_fit.pair_dpsi_data_obj.data_bound,
)
axes[1].set_title("dkappa (iterative)")
plt.tight_layout()
plt.savefig(
    os.path.join("output", "potential_correction_interferometer.png"),
    bbox_inches="tight",
)
plt.close()

"""
__Wrap Up__

For hyper-parameter sampling with a non-linear search, wrap the fits in `al.pc.DpsiSrcInvInterferometerAnalysis`
(one-shot) or `al.pc.IterDpsiSrcInvInterferometerAnalysis` (iterative) — the Bayesian evidence consistently
ranks over-fit and degenerate solutions below recovering ones, so evidence-driven model selection self-protects.

If you use this functionality, please cite Cao et al. 2025
(https://github.com/caoxiaoyue/potential_correction_paper) alongside PyAutoLens.
"""
