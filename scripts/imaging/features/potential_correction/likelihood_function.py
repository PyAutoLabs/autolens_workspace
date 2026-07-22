"""
__Log Likelihood Function: Potential Correction (Gravitational Imaging)__

This script provides a step-by-step guide of the **PyAutoLens** potential-correction `log_likelihood_function`
(the Bayesian evidence of `al.pc.FitDpsiSrcImaging`), which jointly inverts `Imaging` data for a pixelized source
and pixelized corrections $\delta\psi$ to the lensing potential.

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make gravitational imaging less of a "black-box" to users: every operator of the correction formalism
 ($B$, $D_s$, $D_\psi$) is built explicitly as a numpy array below.

If you use the potential-correction functionality in your research, please cite Cao et al. 2025, from whose
`potential_correction` package (https://github.com/caoxiaoyue/lensing_potential_correction) the implementation is
ported; citation materials are provided at https://github.com/caoxiaoyue/potential_correction_paper. The method
builds on Koopmans 2005, Suyu et al. 2009, Vegetti & Koopmans 2009 and Vernardos & Koopmans 2022, and is the
image-plane analogue of the methodology behind the B1938+666 detections of Powell et al. 2025 (Nature Astronomy 9,
1714) and Vegetti et al. 2026.

__Contents__

- **Simplifications:** The choices made to keep this walkthrough small and explicit.
- **Prerequisites:** The likelihood functions this one builds on.
- **Dataset:** Simulate the imaging dataset fitted: a lens whose true mass contains a dark subhalo.
- **Arc Mask:** Mask the data to the arc region where the corrections are constrained.
- **Smooth Starting Model:** The smooth lens model and source the corrections perturb.
- **Dpsi Mesh:** The coarse rectangular mesh the corrections are defined on, paired to the data grid.
- **Dpsi Gradient Operator:** The sparse operator taking mesh corrections to their image-plane gradients.
- **Source Gradients:** The source's brightness gradients at the ray-traced position of every image pixel.
- **PSF Blur Matrix:** The explicit matrix form of the PSF convolution.
- **Dpsi Mapping Matrix:** Combining the three operators into the linear response -B D_s D_psi.
- **Source Inversion Blocks:** The standard pixelized source inversion supplying the source blocks.
- **Joint System:** The block mapping matrix and block-diagonal regularization of the joint inversion.
- **Solve:** Solving the joint linear system for the source and the corrections.
- **Dkappa Map:** Converting the solved corrections into a convergence-correction map.
- **Evidence Terms:** The five terms of the Bayesian evidence, computed explicitly.
- **Verification:** The same number from `al.pc.FitDpsiSrcImaging.log_evidence`.
- **JAX / xp:** The same evidence through the `al.pc.dense_util` xp-API kernels.
- **Wrap Up:** Summary and next steps.

__Simplifications__

To keep every array small enough to inspect, this example uses a low-resolution simulated dataset and a
`RectangularUniform` source mesh (uniform source pixels are simpler to reason about than adaptive meshes). The
corrections are regularized with the `MaternKernel` scheme (`nu=2.5`) used in science analyses — the key result of
Cao et al. 2025 is that the Matern family recovers localised perturbers where classic curvature-penalty schemes
(`aa.reg.CurvatureMask`, also usable here) smear them out. The smooth starting model and source are the truths
used in the simulation, isolating the correction formalism itself.

__Prerequisites__

- `imaging/likelihood_function.ipynb` — the standard imaging likelihood.
- `imaging/features/pixelization/likelihood_function.ipynb` — the pixelized-source evidence, whose terms reappear here.
- `guides/advanced/potential_correction.ipynb` — the user-facing overview of the technique.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import block_diag

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We simulate the dataset in-memory (seeded, so this script is fully reproducible): an `IsothermalSph` lens whose
true mass also contains a $10^{10} M_\odot$ NFW subhalo on the Einstein ring, lensing a compact Gaussian source.
The subhalo is what the corrections will recover.
"""
grid = al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.08, over_sample_size=4)
psf = al.Convolver.from_gaussian(shape_native=(9, 9), sigma=0.08, pixel_scales=0.08)

simulator = al.SimulatorImaging(
    exposure_time=840.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.4),
    subhalo=al.mp.NFWMCRLudlowSph(
        centre=(1.41, 0.0), mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
    ),
)
source_true = al.Galaxy(
    redshift=0.6,
    bulge=al.lp.Gaussian(centre=(0.0, 0.0), intensity=5.0, sigma=0.2),
)

dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)

"""
__Arc Mask__

The corrections respond to the data only through the source's brightness gradients, which vanish away from the
lensed arcs — off-arc pixels carry no information about $\delta\psi$. The data is therefore masked to an
arc-tracing region: `al.pc.util.arc_mask_from` thresholds the signal-to-noise map, drops small islands, dilates
the result and cleans it so every unmasked pixel supports the finite-difference operators built below.
"""
mask_array = al.pc.util.arc_mask_from(
    np.asarray(dataset.signal_to_noise_map.native),
    threshold=3.0,
    ignore_size=10,
    ext_size=3,
)
mask = al.Mask2D(mask=mask_array, pixel_scales=dataset.pixel_scales)
masked_imaging = dataset.apply_mask(mask=mask)

n_data = int(np.count_nonzero(~mask_array))
print(f"unmasked data pixels: n_data = {n_data}")

"""
__Smooth Starting Model__

Potential corrections perturb a smooth starting model — in a real analysis, the maximum-likelihood result of a
standard parametric fit. Here we use the true smooth lens (without the subhalo) and the true source.

The source enters the formalism through a `SrcFactory`, which can evaluate the source's brightness and its
gradients $(\partial S / \partial y, \partial S / \partial x)$ at arbitrary source-plane positions.
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

"""
__Dpsi Mesh__

The corrections are defined on a rectangular mesh a factor coarser than the data grid (here factor 2).
`PairRegularDpsiMesh` builds everything the formalism needs from the data mask alone:

 - `mask_dpsi`: the coarse mesh's mask (a coarse pixel is unmasked only if all its data pixels are);
 - `itp_mat`: a sparse [n_data, n_dpsi] bilinear interpolation matrix from mesh to data grid;
 - `Hx_dpsi`, `Hy_dpsi`: sparse first-derivative operators on the mesh's unmasked pixels;
 - `hamiltonian_dpsi`: the mesh Laplacian, used later to convert $\delta\psi$ to $\delta\kappa$.

Every row of `itp_mat` holds the four bilinear weights of the coarse-mesh box enclosing that data pixel, so its
rows sum to exactly 1.
"""
pair = al.pc.PairRegularDpsiMesh(
    mask_array, pixel_scale=dataset.pixel_scales[0], dpsi_factor=2
)

n_dpsi = int(np.count_nonzero(~pair.mask_dpsi))
print(f"unmasked dpsi mesh pixels: n_dpsi = {n_dpsi}")
print(
    f"itp_mat shape = {pair.itp_mat.shape}, row sums all 1: {np.allclose(pair.itp_mat.sum(axis=1), 1.0)}"
)

"""
__Dpsi Gradient Operator__

The correction $\delta\psi$ deflects rays by its gradient: $\delta\alpha = \nabla \delta\psi$. The sparse operator

$D_\psi = \rm{interleave}(\rm{itp} \cdot H_x, \; \rm{itp} \cdot H_y)$

of shape [2 n_data, n_dpsi] takes the mesh corrections to their (x, y) gradients at every data pixel, with the
per-pixel rows interleaved as $(x_0, y_0, x_1, y_1, ...)$.
"""
dpsi_gradient_matrix = al.pc.util.dpsi_gradient_matrix_from(
    pair.itp_mat, pair.Hx_dpsi, pair.Hy_dpsi
)
print(f"D_psi shape = {dpsi_gradient_matrix.shape}")

"""
__Source Gradients__

A small extra deflection $\delta\alpha$ at an image pixel re-samples the source at a position shifted
by $-\delta\alpha$, changing the observed brightness by $-\nabla S \cdot \delta\alpha$ to first order. We
therefore need the source's gradients at the ray-traced (source-plane) position of every image pixel:

 - ray-trace the masked grid through the smooth lens;
 - evaluate the source's $(\partial S/\partial y, \partial S/\partial x)$ there by central differences
   (`SrcFactory.eval_grad`);
 - pack them into the sparse [n_data, 2 n_data] matrix $D_s$, whose row $i$ holds $(\partial_x S_i, \partial_y S_i)$
   in the columns matching $D_\psi$'s interleaved rows.
"""
traced_grid = masked_imaging.grid.slim - lens_smooth.deflections_yx_2d_from(
    masked_imaging.grid.slim
)
source_gradients = source_start.eval_grad(traced_grid[:, 1], traced_grid[:, 0])
source_gradient_matrix = al.pc.util.source_gradient_matrix_from(source_gradients)
print(f"D_s shape = {source_gradient_matrix.shape}")

"""
__PSF Blur Matrix__

The correction's brightness response is blurred by the telescope PSF like any other emission. The formalism uses
the explicit blur matrix $B$ of shape [n_data, n_data]: column $i$ is the PSF kernel centred on pixel $i$,
restricted to the mask.
"""
psf_matrix = al.pc.util.psf_matrix_from(
    np.asarray(masked_imaging.psf.kernel.native), np.asarray(masked_imaging.mask)
)
print(f"B shape = {psf_matrix.shape}")

"""
__Dpsi Mapping Matrix__

Combining the three operators gives the linear response of the observed image to the mesh corrections
(eq. 8-9 of the potential-correction formalism; Cao et al. 2025):

$\delta d = - B \, D_s \, D_\psi \, \delta\psi$

The minus sign: a positive potential bump deflects rays outward, sampling the source closer to its centre.
"""
dpsi_mapping_matrix = np.asarray(
    -1.0 * psf_matrix @ source_gradient_matrix @ dpsi_gradient_matrix
)
print(f"dpsi mapping matrix shape = {dpsi_mapping_matrix.shape}")

"""
__Source Inversion Blocks__

The joint inversion reconstructs the source simultaneously. Its source blocks come from the standard pixelized
source inversion at the smooth model (see `features/pixelization/likelihood_function.py` for the full
walkthrough): the PSF-operated mapping matrix $F_{\rm src}$ and the source regularization matrix $R_s$.
"""
src_pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=0.6, pixelization=src_pixelization)
tracer = al.Tracer(galaxies=[lens_smooth, source_galaxy])
src_fit = al.FitImaging(
    dataset=masked_imaging.apply_over_sampling(
        over_sample_size_lp=4, over_sample_size_pixelization=4
    ),
    tracer=tracer,
    settings=al.Settings(use_positive_only_solver=True, use_border_relocator=True),
)
src_mapping_matrix = np.asarray(src_fit.inversion.operated_mapping_matrix)
src_regularization_matrix = np.asarray(src_fit.inversion.regularization_matrix)
n_src = src_regularization_matrix.shape[0]
print(f"F_src shape = {src_mapping_matrix.shape}, n_src = {n_src}")

"""
__Joint System__

The joint linear system stacks the two blocks:

 - mapping matrix $M = [\, F_{\rm src} \; | \; -B D_s D_\psi \,]$ of shape [n_data, n_src + n_dpsi];
 - block-diagonal regularization $R = \rm{diag}(R_s, R_\psi)$, where $R_\psi$ is the `MaternKernel` scheme built
   from the dpsi mesh's unmasked pixel positions through the `DpsiLinearObj` adapter (mask-based schemes like
   `CurvatureMask` plug into the same adapter, which exposes both the mesh's `mask` and its pixel positions).

Because the two blocks share one linear solve, the source-vs-corrections covariance is fully accounted for — the
corrections cannot silently absorb source structure the data attributes to the source, and vice versa.
"""
dpsi_regularization = al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5)
dpsi_points = np.vstack([pair.ygrid_dpsi_1d, pair.xgrid_dpsi_1d]).T
dpsi_linear_obj = al.pc.DpsiLinearObj(mask=pair.mask_dpsi, points=dpsi_points)
dpsi_regularization_matrix = dpsi_regularization.regularization_matrix_from(
    linear_obj=dpsi_linear_obj
)

mapping_matrix = np.hstack([src_mapping_matrix, dpsi_mapping_matrix])
regularization_matrix = np.asarray(
    block_diag([src_regularization_matrix, dpsi_regularization_matrix]).toarray()
)
print(f"joint mapping matrix shape = {mapping_matrix.shape}")

"""
__Solve__

With diagonal noise covariance $C^{-1} = \rm{diag}(1/\sigma_i^2)$, the maximum-evidence solution of the joint
system is the standard regularized normal equations (Warren & Dye 2003; Suyu et al. 2006):

$(M^T C^{-1} M + R) \, x = M^T C^{-1} d$

whose solution vector splits into the reconstructed source (first n_src entries) and the corrections
(last n_dpsi entries).
"""
data = np.asarray(masked_imaging.data)
noise = np.asarray(masked_imaging.noise_map)
inv_variance = 1.0 / noise**2

curvature_matrix = mapping_matrix.T @ (mapping_matrix * inv_variance[:, None])
data_vector = mapping_matrix.T @ (inv_variance * data)
curvature_reg_matrix = curvature_matrix + regularization_matrix

solution = np.linalg.solve(curvature_reg_matrix, data_vector)
source_solution = solution[:n_src]
dpsi_solution = solution[n_src:]
model_image = mapping_matrix @ solution

"""
__Dkappa Map__

The corrections' physical meaning is clearest as a convergence correction,
$\delta\kappa = \frac{1}{2}\nabla^2 \delta\psi$, via the mesh Laplacian. A dark subhalo missing from the smooth
model appears as a positive $\delta\kappa$ peak at its position.
"""
dkappa = pair.hamiltonian_dpsi @ dpsi_solution
peak = int(np.argmax(dkappa))
print(
    f"dkappa peak at (y, x) = ({pair.ygrid_dpsi_1d[peak]:.2f}, {pair.xgrid_dpsi_1d[peak]:.2f})"
    f" — true subhalo at (1.41, 0.00)"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
al.pc.visualize.imshow_masked_data(
    dpsi_solution, pair.mask_dpsi, ax=axes[0], origin="upper", extent=pair.data_bound
)
axes[0].set_title("dpsi")
al.pc.visualize.imshow_masked_data(
    dkappa, pair.mask_dpsi, ax=axes[1], origin="upper", extent=pair.data_bound
)
axes[1].set_title("dkappa")
plt.tight_layout()
plt.show()
plt.close()

"""
__Evidence Terms__

The Bayesian evidence of the joint inversion has five terms (Suyu et al. 2006 eq. 19; Cao et al. 2025):

 1. the noise normalization $-\frac{1}{2}\sum_i \log(2\pi\sigma_i^2)$;
 2. the Occam term $-\frac{1}{2}\log\det(M^T C^{-1} M + R)$, penalising flexible models;
 3. the regularization normalizations $+\frac{1}{2}[\log\det R_s + \log\det R_\psi]$;
 4. the regularization penalty of the solution $-\frac{1}{2} x^T R x$;
 5. the $\chi^2$ of the fit $-\frac{1}{2}\sum_i (d_i - m_i)^2/\sigma_i^2$.

Terms 2-4 are what allow the evidence to set both regularization strengths objectively: stronger regularization
lowers the Occam term but raises the penalty and chi-squared, and the evidence peaks at the balance.
"""
noise_term = -0.5 * float(np.sum(np.log(2.0 * np.pi * noise**2)))
occam_term = -0.5 * float(np.linalg.slogdet(curvature_reg_matrix)[1])
reg_norm_term = 0.5 * (
    float(np.linalg.slogdet(src_regularization_matrix)[1])
    + float(np.linalg.slogdet(dpsi_regularization_matrix)[1])
)
reg_penalty_term = -0.5 * float(solution @ regularization_matrix @ solution)
chi_squared_term = -0.5 * float(np.sum(((data - model_image) / noise) ** 2))

log_evidence = (
    noise_term + occam_term + reg_norm_term + reg_penalty_term + chi_squared_term
)

print(f"noise term          = {noise_term:.6e}")
print(f"occam term          = {occam_term:.6e}")
print(f"reg normalizations  = {reg_norm_term:.6e}")
print(f"reg penalty         = {reg_penalty_term:.6e}")
print(f"chi squared term    = {chi_squared_term:.6e}")
print(f"log evidence        = {log_evidence:.8e}")

"""
__Verification__

`al.pc.FitDpsiSrcImaging` performs exactly the steps above. Its `log_evidence` must equal our explicit
calculation (we preload the source blocks so both use the identical source inversion).
"""
fit = al.pc.FitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=2), regularization=dpsi_regularization
    ),
    src_pixelization=src_pixelization,
    preloads={
        "src_map_mat": src_mapping_matrix,
        "src_reg_mat": src_regularization_matrix,
    },
)

print(f"al.pc.FitDpsiSrcImaging.log_evidence = {fit.log_evidence:.8e}")
assert np.isclose(fit.log_evidence, log_evidence, rtol=1e-8)

"""
__JAX / xp__

Every dense step above is also available through `al.pc.dense_util`, whose kernels follow the PyAuto `xp`
convention: written once with `xp=np` (what we just did by hand), and runnable under `xp=jax.numpy` for
jit-compiled, accelerator-ready execution — see
`autolens_workspace_test/scripts/jax_likelihood_functions/imaging/potential_correction.py` for the numerical
agreement checks.

    from autolens.potential_correction import dense_util

    result = dense_util.log_evidence_joint_dense_from(
        data, noise, mapping_matrix, src_regularization_matrix, dpsi_regularization_matrix, xp=np,  # or jax.numpy
    )

__Wrap Up__

We built the potential-correction evidence from its raw ingredients: the dpsi mesh and its interpolation to the
data grid, the gradient operators, the PSF blur matrix, the joint block system and its five evidence terms —
and verified `al.pc.FitDpsiSrcImaging` reproduces the number exactly.

In a science analysis the regularization hyper-parameters are sampled with a non-linear search through
`al.pc.DpsiSrcInvAnalysis` (one-shot) or refined with the iterative Levenberg-Marquardt engine
`al.pc.IterFitDpsiSrcImaging` — see `guides/advanced/potential_correction.py`.

If you use this functionality, please cite Cao et al. 2025
(https://github.com/caoxiaoyue/potential_correction_paper) alongside PyAutoLens.
"""
