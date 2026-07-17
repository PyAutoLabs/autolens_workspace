"""
__Log Likelihood Function: Potential Correction (Interferometer)__

This script provides a step-by-step guide of the **PyAutoLens** visibility-space potential-correction
`log_likelihood_function` (the Bayesian evidence of `al.pc.FitDpsiSrcInterferometer`), which jointly inverts
`Interferometer` data for a pixelized source and pixelized corrections $\delta\psi$ to the lensing potential.

This script has the following aims:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand the
 likelihood function (including references to the previous literature from which it is defined) without having to
 write large quantities of text and equations.

 - To make visibility-space gravitational imaging less of a "black-box" to users: every operator of the correction
 formalism ($D_s$, $D_\psi$, the joint response and both curvature routes) is built explicitly as a numpy array below.

If you use the potential-correction functionality in your research, please cite Cao et al. 2025, from whose
`potential_correction` package (https://github.com/caoxiaoyue/lensing_potential_correction) the implementation is
ported; citation materials are provided at https://github.com/caoxiaoyue/potential_correction_paper. The method
builds on Koopmans 2005, Suyu et al. 2009 and Vegetti & Koopmans 2009; the visibility-space formulation is the
methodology behind the JVAS B1938+666 detections of Powell et al. 2025 (Nature Astronomy 9, 1714) and
Vegetti et al. 2026.

__Contents__

- **Simplifications:** The choices made to keep this walkthrough small and explicit.
- **Prerequisites:** The likelihood functions this one builds on.
- **Dataset:** Simulate the interferometer dataset fitted: a lens whose true mass contains a dark subhalo.
- **Source Inversion Blocks:** The standard pixelized source inversion supplying the source blocks.
- **Dpsi Mesh:** The arc-restricted coarse mesh the corrections are defined on.
- **Correction Response:** The real-space response G = -D_s D_psi of the image to the corrections.
- **Joint Response:** The joint real-space response A = [f | G] — no PSF matrix: the measurement operator here
  is the non-uniform Fourier transform.
- **Dense Route:** The visibility-space normal equations built from the explicitly transformed response.
- **Sparse (w-tilde) Route:** The same curvature and data vector without ever forming the transformed response.
- **Solve:** Solving the joint linear system for the source and the corrections.
- **Dkappa Map:** Converting the solved corrections into a convergence-correction map.
- **Evidence Terms:** The five terms of the Bayesian evidence, computed explicitly.
- **Verification:** The same number from `al.pc.FitDpsiSrcInterferometer.log_evidence`, both routes.
- **JAX / xp:** The xp-API kernels of the same computation.
- **Wrap Up:** Summary and next steps.

__Simplifications__

To keep every array small enough to inspect, this example uses a low-resolution simulated dataset with only 1000
visibilities — small enough that the dense reference route (which materializes the [2 n_vis, n_par] transformed
response) fits in memory alongside the sparse route. The source is reconstructed on a `RectangularUniform` mesh
(uniform source pixels are simpler to reason about than adaptive meshes) and the corrections are regularized with
the `MaternKernel` scheme (`nu=2.5`) used in science analyses. The smooth starting model and source are the truths
used in the simulation, isolating the correction formalism itself.

__Prerequisites__

- `interferometer/features/pixelization/likelihood_function.ipynb` — the visibility-space pixelized-source
  evidence, whose terms reappear here.
- `imaging/features/potential_correction/likelihood_function.ipynb` — the imaging analogue of this walkthrough,
  where the measurement operator is the PSF-blur matrix instead of the Fourier transform.
- `interferometer/features/potential_correction/start_here.ipynb` — the user-facing example of this feature.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import block_diag

import autoarray as aa
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We simulate the dataset in-memory (seeded, so this script is fully reproducible): random uv coverage, an
`Isothermal` lens whose true mass also contains a $10^{10} M_\odot$ NFW subhalo on the Einstein ring, and a
compact double-Gaussian source. The subhalo is what the corrections will recover.

Unlike imaging, the real-space mask must stay a **filled circle**: it defines the rectangular extent of the FFTs
inside the sparse operator.
"""
rng = np.random.default_rng(1)
uv_wavelengths = rng.uniform(-3.0e5, 3.0e5, size=(1000, 2))

real_space_mask = al.Mask2D.circular(shape_native=(64, 64), pixel_scales=0.1, radius=2.6)
grid = al.Grid2D.from_mask(mask=real_space_mask)

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=0.02,
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
        centre=(1.41, 0.0), mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
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

dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)
dataset = al.Interferometer(
    data=dataset.data,
    noise_map=dataset.noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
)
dataset = dataset.apply_sparse_operator()

n_full = int(np.count_nonzero(~np.asarray(real_space_mask)))
print(f"visibilities: {uv_wavelengths.shape[0]}, real-space mask pixels: n_full = {n_full}")

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
__Source Inversion Blocks__

The joint inversion reconstructs the source simultaneously. Its source blocks come from the standard pixelized
visibility-space source inversion at the smooth model (see
`interferometer/features/pixelization/likelihood_function.py` for the full walkthrough): the mapper's real-space
mapping matrix $f$ of shape [n_full, n_src] and the source regularization matrix $R_s$.

Note that $f$ is a **real-space** matrix — unlike imaging, where the source block is PSF-convolved before entering
the joint system, here the measurement operator (the Fourier transform) is applied to the whole joint response at
once, further below.
"""
src_pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=3.8),
)
source_galaxy = al.Galaxy(redshift=1.0, pixelization=src_pixelization)
src_fit = al.FitInterferometer(
    dataset=dataset,
    tracer=al.Tracer(galaxies=[lens_smooth, source_galaxy]),
    settings=aa.Settings(use_positive_only_solver=True, use_border_relocator=True),
)
mapper = src_fit.inversion.linear_obj_list[0]
src_mapping_matrix = np.asarray(mapper.mapping_matrix)
src_regularization_matrix = np.asarray(src_fit.inversion.regularization_matrix)
n_src = src_regularization_matrix.shape[0]
print(f"f shape = {src_mapping_matrix.shape}, n_src = {n_src}")

"""
__Dpsi Mesh__

The corrections are only constrained where the lensed arcs are, so their mesh is restricted to an arc-tracing
sub-mask of the real-space mask (built here from the smooth-model image geometry). `PairRegularDpsiMesh` then
builds everything the formalism needs on a mesh a factor coarser than the data grid:

 - `mask_data` / `mask_dpsi`: the cleaned arc mask on the data grid and its coarse-mesh counterpart;
 - `itp_mat`: a sparse bilinear interpolation matrix from mesh to data grid (rows sum to exactly 1);
 - `Hx_dpsi`, `Hy_dpsi`: sparse first-derivative operators on the mesh's unmasked pixels;
 - `hamiltonian_dpsi`: the mesh Laplacian, used later to convert $\delta\psi$ to $\delta\kappa$.
"""
tracer_smooth = al.Tracer(galaxies=[lens_smooth, source_true])
arc_image = np.asarray(tracer_smooth.image_2d_from(grid=grid).native)
arc_mask = al.pc.util.arc_mask_from(
    arc_image / (0.05 * arc_image.max()), threshold=3.0, ignore_size=10, ext_size=3
)
dpsi_mask = ~((~arc_mask) & (~np.asarray(real_space_mask)))

pair = al.pc.PairRegularDpsiMesh(dpsi_mask, pixel_scale=0.1, dpsi_factor=2)
n_sub = int(np.count_nonzero(~pair.mask_data))
n_dpsi = int(np.count_nonzero(~pair.mask_dpsi))
print(f"arc pixels: n_sub = {n_sub}, dpsi mesh pixels: n_dpsi = {n_dpsi}")

"""
__Correction Response__

As in the imaging walkthrough, a correction $\delta\psi$ perturbs the observed image via the source's brightness
gradients at the ray-traced position of every (arc) image pixel:

$\delta I = - D_s \, D_\psi \, \delta\psi \equiv G \, \delta\psi$

 - $D_\psi = \rm{interleave}(\rm{itp} \cdot H_x, \; \rm{itp} \cdot H_y)$, of shape [2 n_sub, n_dpsi], takes the
   mesh corrections to their (x, y) gradients at every arc pixel;
 - $D_s$, of shape [n_sub, 2 n_sub], holds the source's gradients at the traced positions;
 - the minus sign: a positive potential bump deflects rays outward, sampling the source closer to its centre.

The arc rows are then scattered into the full real-space row space (zero response off the arcs), giving the
real-space correction response $G$ of shape [n_full, n_dpsi]. There is **no PSF-blur matrix**: blurring by the
instrument response happens in visibility space, through the Fourier transform below.
"""
dpsi_gradient_matrix = al.pc.util.dpsi_gradient_matrix_from(
    pair.itp_mat, pair.Hx_dpsi, pair.Hy_dpsi
)

traced_grid = np.asarray(
    dataset.grid.slim - lens_smooth.deflections_yx_2d_from(dataset.grid.slim)
)
full_index = np.full(np.asarray(real_space_mask).shape, -1, dtype=int)
full_index[~np.asarray(real_space_mask)] = np.arange(n_full)
rows_in_full = full_index[~pair.mask_data]

source_gradients = source_start.eval_grad(
    traced_grid[rows_in_full, 1], traced_grid[rows_in_full, 0]
)
source_gradient_matrix = al.pc.util.source_gradient_matrix_from(source_gradients)

G_sub = np.asarray((-1.0 * source_gradient_matrix @ dpsi_gradient_matrix).todense())
G = np.zeros((n_full, n_dpsi))
G[rows_in_full] = G_sub
print(f"G shape = {G.shape}")

"""
__Joint Response__

The joint real-space response stacks the two blocks, $A = [\, f \; | \; G \,]$, alongside the block-diagonal
regularization $R = \rm{diag}(R_s, R_\psi)$, where $R_\psi$ is the `MaternKernel` scheme built from the dpsi
mesh's unmasked pixel positions through the `DpsiLinearObj` adapter.

Because the two blocks share one linear solve, the source-vs-corrections covariance is fully accounted for — the
corrections cannot silently absorb structure the data attributes to the source, and vice versa.
"""
dpsi_regularization = al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5)
dpsi_points = np.vstack([pair.ygrid_dpsi_1d, pair.xgrid_dpsi_1d]).T
dpsi_linear_obj = al.pc.DpsiLinearObj(mask=pair.mask_dpsi, points=dpsi_points)
dpsi_regularization_matrix = np.asarray(
    dpsi_regularization.regularization_matrix_from(linear_obj=dpsi_linear_obj)
)

A = np.hstack([src_mapping_matrix, G])
regularization_matrix = np.asarray(
    block_diag([src_regularization_matrix, dpsi_regularization_matrix]).toarray()
)
print(f"joint real-space response A shape = {A.shape}")

"""
__Dense Route__

The measurement operator is the non-uniform Fourier transform $T$, taking any real-space image to model
visibilities. The dense reference route materializes the transformed joint response $T(A)$, stacks its real and
imaginary parts row-wise into $M$ of shape [2 n_vis, n_par], and forms the standard regularized normal equations
(Warren & Dye 2003; Suyu et al. 2006) with diagonal noise covariance $C^{-1}$:

curvature $F = M^T C^{-1} M$, data vector $D = M^T C^{-1} d$.

This is exact but scales with the visibility count — fine at 1000 visibilities, impossible at the $10^5$-$10^8$
of real ALMA/VLBI datasets.
"""
transformed = dataset.transformer.transform_mapping_matrix(A)
M = np.vstack([np.real(transformed), np.imag(transformed)])

data = np.asarray(dataset.data)
noise = np.asarray(dataset.noise_map)
stacked_data = np.concatenate([data.real, data.imag])
stacked_inv_variance = 1.0 / np.concatenate([noise.real, noise.imag]) ** 2

curvature_matrix = M.T @ (M * stacked_inv_variance[:, None])
data_vector = M.T @ (stacked_inv_variance * stacked_data)
print(f"dense route: M shape = {M.shape}, curvature shape = {curvature_matrix.shape}")

"""
__Sparse (w-tilde) Route__

The production route computes the identical $F$ and $D$ without ever forming $T(A)$ (Powell et al. 2021's
visibility-space w-tilde formalism):

 - the curvature is $F = A^T \, (T^H C^{-1} T) \, A$, where the operator $T^H C^{-1} T$ is a **convolution** in
   real space — applied by FFTs on the mask's rectangular extent, so the cost depends on the number of mask
   pixels, not the number of visibilities;
 - the data vector is $D = A^T \tilde{d}$, where $\tilde{d} = T^H C^{-1} d$ is the **dirty image** of the
   visibilities — computed once when `apply_sparse_operator` is called.

Both are available from the dataset's sparse operator; we verify they equal the dense-route matrices.
"""
data_vector_sparse = A.T @ np.asarray(dataset.sparse_operator.dirty_image)
print(
    f"sparse data vector == dense data vector: "
    f"{np.allclose(data_vector_sparse, data_vector, rtol=1e-6)}"
)

"""
The sparse curvature is assembled from the COO triplets of $A$ (the machinery `al.pc.FitDpsiSrcInterferometer`
uses internally); rather than repeat that plumbing here, we take the fit's sparse-route curvature below and verify
it against our dense $F$ at the end. For this walkthrough we continue with the dense-route matrices, which we have
just shown are the same thing.

__Solve__

The maximum-evidence solution of the joint system:

$(F + R) \, x = D$

splits into the reconstructed source (first n_src entries) and the corrections (last n_dpsi entries).
"""
curvature_reg_matrix = curvature_matrix + regularization_matrix
solution = np.linalg.solve(curvature_reg_matrix, data_vector)
source_solution = solution[:n_src]
dpsi_solution = solution[n_src:]

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

The Bayesian evidence of the joint inversion has five terms (Suyu et al. 2006 eq. 19; Cao et al. 2025), with the
visibility-space specifics that the noise normalization and $\chi^2$ run over the real **and** imaginary parts:

 1. the noise normalization $-\frac{1}{2}\sum_i [\log(2\pi\sigma_{R,i}^2) + \log(2\pi\sigma_{I,i}^2)]$;
 2. the Occam term $-\frac{1}{2}\log\det(F + R)$, penalising flexible models;
 3. the regularization normalizations $+\frac{1}{2}[\log\det R_s + \log\det R_\psi]$;
 4. the regularization penalty of the solution $-\frac{1}{2} x^T R x$;
 5. the $\chi^2$ of the fit, $-\frac{1}{2}\sum_i |d_i - m_i|^2/\sigma_i^2$, where the model visibilities
    $m = T(A x)$ cost one forward transform of the reconstructed real-space image — the only place a transform of
    a model is ever needed in the sparse route.

Terms 2-4 are what allow the evidence to set both regularization strengths objectively: stronger regularization
lowers the Occam term but raises the penalty and chi-squared, and the evidence peaks at the balance.
"""
noise_term = -0.5 * float(aa.util.fit.noise_normalization_complex_from(noise_map=noise))
occam_term = -0.5 * float(np.linalg.slogdet(curvature_reg_matrix)[1])
reg_norm_term = 0.5 * (
    float(np.linalg.slogdet(src_regularization_matrix)[1])
    + float(np.linalg.slogdet(dpsi_regularization_matrix)[1])
)
reg_penalty_term = -0.5 * float(solution @ regularization_matrix @ solution)

model_image = aa.Array2D(values=A @ solution, mask=real_space_mask)
model_visibilities = np.asarray(dataset.transformer.visibilities_from(image=model_image))
residual = data - model_visibilities
chi_squared_term = -0.5 * float(
    np.sum((residual.real / noise.real) ** 2) + np.sum((residual.imag / noise.imag) ** 2)
)

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

`al.pc.FitDpsiSrcInterferometer` performs exactly the steps above, through either route. Its `log_evidence` must
equal our explicit calculation for both (we preload the source blocks so all three use the identical source
inversion), and its sparse-route curvature must equal our dense-route $F$.
"""
dpsi_pixelization = al.pc.DpsiPixelization(
    mesh=al.pc.RegularDpsiMesh(factor=2), regularization=dpsi_regularization
)
preloads = {"src_mapper": mapper, "src_reg_mat": src_regularization_matrix}

fit_sparse = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    dpsi_mask=dpsi_mask,
    use_sparse_operator=True,
    preloads=preloads,
)
fit_dense = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    dpsi_mask=dpsi_mask,
    use_sparse_operator=False,
    preloads=preloads,
)

print(f"sparse-route log_evidence = {fit_sparse.log_evidence:.8e}")
print(f"dense-route  log_evidence = {fit_dense.log_evidence:.8e}")

assert np.allclose(np.asarray(fit_sparse.curvature_matrix), curvature_matrix, rtol=1e-6)
assert np.isclose(fit_dense.log_evidence, log_evidence, rtol=1e-8)
assert np.isclose(fit_sparse.log_evidence, log_evidence, rtol=1e-6)

"""
__JAX / xp__

The iterative refinement engine (`al.pc.IterFitDpsiSrcInterferometer`, which re-ray-traces through the corrected
lens and re-solves) runs its Levenberg-Marquardt kernels through the PyAuto `xp` convention: written once with
`xp=np` (what we just did by hand), and runnable under `xp=jax.numpy` for jit-compiled, accelerator-ready
execution. The sparse operator's FFT machinery is JAX-accelerated in all cases — see
`autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/potential_correction.py` for the
numerical agreement checks.

__Wrap Up__

We built the visibility-space potential-correction evidence from its raw ingredients: the arc-restricted dpsi
mesh, the gradient operators, the joint real-space response, the dense and sparse (w-tilde) routes to the same
normal equations, and the five evidence terms — and verified `al.pc.FitDpsiSrcInterferometer` reproduces the
number exactly through both routes.

In a science analysis the regularization hyper-parameters are sampled with a non-linear search through
`al.pc.DpsiSrcInvInterferometerAnalysis` (one-shot) or refined with the iterative engine — see `start_here.py`
in this folder for the certified end-to-end recipe.

If you use this functionality, please cite Cao et al. 2025
(https://github.com/caoxiaoyue/potential_correction_paper) alongside PyAutoLens.
"""
