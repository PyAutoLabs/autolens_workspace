"""
__Log Likelihood Function: Multi Gaussian Expansion (Interferometer)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit
`Interferometer` data with a multi-Gaussian expansion (MGE) source: a `Basis` of many linear `Gaussian`
profiles whose `intensity` values are solved for analytically via a linear inversion.

The visibility-plane linear inversion is **mathematically identical** to the single-component linear light
profile case (see `interferometer/features/linear_light_profiles/likelihood_function.py`). The only
difference is the number of columns in the mapping matrix — one column per Gaussian in the basis instead
of one column total. The data vector `D` and curvature matrix `F` therefore have dimensions equal to the
basis size, and the solved `s = F^{-1} D` gives one `intensity` per Gaussian.

**Role swap vs the imaging MGE example:** the imaging script's basis is on the lens galaxy. Here it's on
the source galaxy, because interferometer data does not contain lens light. The source bulge image is
evaluated on the *ray-traced* source-plane grid; the rest of the inversion machinery is unchanged.

For interferometer data this is now practical thanks to the JAX-native NUFFT `nufftax`
(https://github.com/GragasLab/nufftax) — every Gaussian's NUFFT happens inside the same jit/vmap pipeline
as the rest of the model, so the per-iteration NUFFT cost scales linearly with the basis size and is
amortised on the GPU.

__Prerequisites__

The MGE likelihood function builds on:

 - `interferometer/likelihood_function.ipynb` — the standard interferometer parametric likelihood
   function (NUFFT of a real-space image, visibility-plane $\\chi^2$).
 - `interferometer/features/linear_light_profiles/likelihood_function.ipynb` — the single-component
   visibility-plane linear inversion (data vector, curvature matrix, positive-only solver). The MGE is
   a direct generalisation to N-component bases.

This script repeats just enough setup that you can follow it without rereading those two — but if anything
is unclear, those are the places to look first.

__Contents__

- **Prerequisites:** Reading order before this script.
- **Mask:** Define the `real_space_mask` which sets the grid the strong lens is evaluated on.
- **Dataset:** Load and plot the strong lens `Interferometer` dataset using `TransformerNUFFT` (nufftax).
- **Lens Galaxy:** A mass-only lens galaxy (no light — interferometer convention).
- **Source Galaxy MGE Basis:** Build the linear `Gaussian` basis used as the source bulge.
- **Ray Tracing:** Image-plane to source-plane grid.
- **Mapping Matrix:** Real-space mapping matrix — one column per Gaussian in the basis.
- **Transformed Mapping Matrix ($f$):** NUFFT each column to give the visibility-space mapping matrix.
- **Data Vector (D):** Compute $D$ from the transformed mapping matrix, visibilities, and noise map.
- **Curvature Matrix (F):** Compute $F$ separately for real and imaginary components, then sum.
- **Reconstruction (Positive-Negative):** Solve $s = F^{-1} D$ via NumPy.
- **Reconstruction (Positive Only):** Solve with the fast non-negative least squares (`fnnls`) algorithm.
- **Visibilities Reconstruction:** Map $s$ back to visibility space.
- **Likelihood Function:** Visibility-plane $\\chi^2$ and noise normalization.
- **Chi Squared:** Sum chi-squared contributions over real and imaginary components.
- **Noise Normalization Term:** The fixed noise normalization term.
- **Calculate The Log Likelihood:** Combine into the final log likelihood.
- **Fit:** Cross-check via `FitInterferometer`.
- **Lens Modeling:** How this likelihood is sampled in the full Nautilus fit.
- **Wrap Up:** Summary and next steps.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Mask__

We define the `real_space_mask` which defines the grid the image of the strong lens is evaluated on.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load and plot the strong lens `Interferometer` dataset `simple` from .fits files using `TransformerNUFFT`
backed by `nufftax`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Lens Galaxy__

For interferometer data the lens galaxy's optical/IR emission is typically below detection, so we model
only its mass:
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)

"""
__Source Galaxy MGE Basis__

We build a `Basis` of 5 linear `Gaussian` profiles for the source bulge, all sharing the same centre and
`ell_comps`, with `sigma` values spanning 0.01" to the mask radius in log-spaced increments.

Each Gaussian is a linear light profile — its `intensity` is solved for analytically via the inversion
below. Internally each linear profile carries `intensity=1.0`, which the inversion later rescales.
"""
total_gaussians = 5
log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []
for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        sigma=10 ** log10_sigma_list[i],
    )
    bulge_gaussian_list.append(gaussian)

source_bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)
source_galaxy = al.Galaxy(redshift=1.0, bulge=source_bulge)

"""
__Ray Tracing__

To perform lensing calculations we ray-trace every 2d (y,x) coordinate $\\theta$ from the image-plane to
its (y,x) source-plane coordinate $\\beta$ using the summed deflection angles $\\alpha$ of the mass
profiles:

 $\\beta = \\theta - \\alpha(\\theta)$

For interferometer data the only grid we need to ray-trace is the real-space grid associated with the
`real_space_mask`.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

# A list of every grid (e.g. image-plane, source-plane); we only need the source-plane grid (index -1).
traced_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.lp)[-1]

aplt.plot_grid(grid=traced_grid, title="Traced Source-Plane Grid")

"""
__Mapping Matrix__

For interferometer MGE modeling, the mapping matrix has one column per Gaussian in the basis. Each column
holds the real-space image of one Gaussian, evaluated on the ray-traced source-plane grid with
`intensity=1.0` internally. The lensing is already baked in because we evaluated on the traced grid.

The dimensions are `(total_real_space_pixels, total_gaussians)`.
"""
lp_linear_func_source = al.LightProfileLinearObjFuncList(
    grid=traced_grid,
    blurring_grid=None,
    psf=None,
    light_profile_list=bulge_gaussian_list,
    regularization=None,
)

mapping_matrix = lp_linear_func_source.mapping_matrix

print("Mapping matrix shape (real-space pixels, total Gaussians):")
print(mapping_matrix.shape)

"""
A 2D plot of the `mapping_matrix` shows each Gaussian's lensed source-plane image as one column.
"""
plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.colorbar()
plt.title("Real-space mapping matrix — one column per Gaussian")
plt.show()
plt.close()

"""
__Transformed Mapping Matrix ($f$)__

Every column of the real-space `mapping_matrix` must be NUFFT'd to the uv-plane. The result is the
`transformed_mapping_matrix` — a *complex-valued* matrix with dimensions
`(total_visibilities, total_gaussians)`.

For an N-Gaussian MGE this matrix has N columns. The inversion's job is to find the N scalars that, when
multiplied by their respective columns and summed, best fit the observed visibilities.

This NUFFT-each-column operation is exactly what nufftax made fast. The whole pipeline (ray-trace →
source-plane Gaussian images → `transform_mapping_matrix` → linear inversion) is JIT-compilable under JAX.
"""
transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

print("Transformed mapping matrix shape (visibilities, total Gaussians):")
print(transformed_mapping_matrix.shape)

"""
Plot the real and imaginary components of the transformed mapping matrix. Each row is one observed
visibility; the values across that row are the per-unit-intensity contributions of each Gaussian at that
uv point.
"""
plt.imshow(
    transformed_mapping_matrix.real,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.title("Re(f) — real component of transformed mapping matrix")
plt.show()
plt.close()

plt.imshow(
    transformed_mapping_matrix.imag,
    aspect=(transformed_mapping_matrix.shape[1] / transformed_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.title("Im(f) — imaginary component of transformed mapping matrix")
plt.show()
plt.close()

"""
Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587) (hereafter WD03) introduce the linear inversion
formalism used to compute the intensity values of the linear light profiles. WD03 indexes the transformed
mapping matrix as $f_{ij}$ where $i$ maps over all $I$ linear basis components and $j$ maps over all $J$
visibilities. For our MGE, $I = $ `total_gaussians`.

The indexing of the `transformed_mapping_matrix` array is reversed compared to the WD03 convention (rows
are visibilities, columns are basis components).

__Data Vector (D)__

To solve for the per-Gaussian intensities we pose the problem as a linear inversion.

The `data_vector`, $D$, has dimensions `(total_gaussians,)`. In WD03 the data vector is given by:

 $\\vec{D}_{i} = \\sum_{\\rm  j=1}^{J} f_{ij}\\, d_{j} / \\sigma_{j}^2 \\, ,$

where $d_j$ are the observed visibility values, $\\sigma_j^2$ are the visibility variances, and the sum
runs over real and imaginary components. The interferometer helper handles the real/imaginary split
internally.
"""
data_vector = (
    al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=dataset.data,
        noise_map=dataset.noise_map,
    )
)

print("Data Vector D shape:")
print(data_vector.shape)

"""
__Curvature Matrix (F)__

The `curvature_matrix` $F$ has dimensions `(total_gaussians, total_gaussians)`.

In WD03 the curvature matrix is given by:

 ${F}_{ik} = \\sum_{\\rm  j=1}^{J} f_{ij}\\, f_{kj} / \\sigma_{j}^2 \\, .$

Because visibilities (and therefore $f$) are complex-valued, the curvature is computed separately for the
real and imaginary parts and summed.

For an N-Gaussian MGE $F$ is an NxN matrix; the off-diagonal entries $F_{ik}$ quantify how much Gaussians
$i$ and $k$ overlap in the visibility plane. Adjacent Gaussians (similar sigma) have large overlaps; very
different sigmas overlap less.
"""
real_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.real,
    noise_map=dataset.noise_map.real,
)

imag_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=transformed_mapping_matrix.imag,
    noise_map=dataset.noise_map.imag,
)

curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)

print("Curvature Matrix F shape:")
print(curvature_matrix.shape)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.title("Curvature Matrix F")
plt.show()
plt.close()

"""
__Reconstruction (Positive-Negative)__

The chi-squared minimised by the inversion is:

$\\chi^2 = \\sum_{\\rm  j=1}^{J} \\bigg[ \\frac{(\\sum_{\\rm  i=1}^{I} s_{i} f_{ij}) - d_{j}}{\\sigma_{j}} \\bigg]^2$

Where $s_i$ is the solved-for `intensity` of the $i$-th Gaussian. The solution is given by (equation 5
WD03):

 $s = F^{-1} D$

For an MGE this often returns negative `intensity` values for some Gaussians — the "ringing" pattern
where alternating Gaussians take large positive/negative values. This is unphysical.
"""
reconstruction_positive_negative = np.linalg.solve(curvature_matrix, data_vector)

print("Reconstruction s (positive-negative solver, first 5):")
print(reconstruction_positive_negative[:5])
print(
    f"  number of negative entries: {(reconstruction_positive_negative < 0).sum()} / {total_gaussians}"
)

"""
__Reconstruction (Positive Only)__

The linear algebra can be solved with the constraint that all `intensity` values are positive. The naive
approach is `scipy.optimize.nnls`, which is iterative and works directly on the transformed mapping matrix
— slow for many Gaussians.

The source code therefore uses a "fast nnls" algorithm — an adaptation of:
  https://github.com/jvendrow/fnnls

`fnnls` uses `data_vector` $D$ and `curvature_matrix` $F$ (not the full mapping matrix), which makes it
much faster than scipy's nnls. This is essential for MGE because the basis size can be large.
"""
reconstruction = al.util.inversion.reconstruction_positive_only_from(
    data_vector=data_vector,
    curvature_reg_matrix=curvature_matrix,  # ignore the _reg_ tag in this guide
)

print("Reconstruction s (positive-only solver, first 5):")
print(reconstruction[:5])
print(f"  number of zero entries: {(reconstruction == 0).sum()} / {total_gaussians}")

"""
__Visibilities Reconstruction__

Using the reconstructed `intensity` values we can map the reconstruction back to the visibility plane via
the `transformed_mapping_matrix`, producing the model visibilities.

This sums the per-Gaussian visibility-plane contributions weighted by their solved-for intensities.
"""
mapped_reconstructed_visibilities = (
    al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        reconstruction=reconstruction,
    )
)

mapped_reconstructed_visibilities = al.Visibilities(
    visibilities=mapped_reconstructed_visibilities
)

aplt.plot_grid(
    grid=mapped_reconstructed_visibilities.in_grid,
    title="Model Visibilities (MGE source)",
)


"""
__Likelihood Function__

We now quantify the goodness-of-fit of our MGE source reconstruction.

The likelihood function for parametric galaxy modeling, even with linear light profile bases, consists of
two terms in the visibility plane:

 $-2 \\mathrm{ln} \\, \\epsilon = \\chi^2 + \\sum_{\\rm  j=1}^{J} { \\mathrm{ln}} \\left [2 \\pi (\\sigma_j)^2 \\right]  \\, .$

(Note: for a *pixelization* there are additional regularisation terms. The MGE inversion does not use
regularisation — its smoothness comes from the basis-function form, not from a regularisation matrix.)

__Chi Squared__

The first term is a $\\chi^2$ statistic computed in the visibility plane:

 - `model_data` = `mapped_reconstructed_visibilities`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = `normalized_residual_map` ** 2.0
 - `chi_squared` = sum(`chi_squared_map`)

Visibilities are complex-valued, so we split into real and imaginary components, compute $\\chi^2$ for
each, and sum.
"""
model_visibilities = mapped_reconstructed_visibilities

residual_map = dataset.data - model_visibilities

chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2

chi_squared_real = np.sum(chi_squared_map_real)
chi_squared_imag = np.sum(chi_squared_map_imag)
chi_squared = chi_squared_real + chi_squared_imag

print(f"chi_squared (real + imag) = {chi_squared}")

"""
__Noise Normalization Term__

The likelihood function assumes the visibility data consists of independent Gaussian noise on every
visibility (real and imaginary parts treated independently).
"""
noise_normalization_real = float(
    np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
)
noise_normalization_imag = float(
    np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
)
noise_normalization = noise_normalization_real + noise_normalization_imag

"""
__Calculate The Log Likelihood__

Combine the two terms to compute the `log_likelihood`:
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(f"log_likelihood (figure of merit) = {figure_of_merit}")


"""
__Fit__

The exact same likelihood evaluation is performed inside the `FitInterferometer` object. We construct one,
print its `figure_of_merit`, and confirm it matches the value we computed by hand above.
"""
fit = al.FitInterferometer(dataset=dataset, tracer=tracer)
print(f"FitInterferometer.figure_of_merit = {fit.figure_of_merit}")

aplt.subplot_fit_interferometer(fit=fit)

"""
The fit contains an `Inversion` object, which handles all the linear algebra we have covered in this
script.
"""
print(fit.inversion)
print(f"data_vector shape:    {fit.inversion.data_vector.shape}")
print(f"curvature_matrix shape: {fit.inversion.curvature_matrix.shape}")
print(f"reconstruction shape:   {fit.inversion.reconstruction.shape}")

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm `nautilus`
(https://github.com/johannesulf/nautilus); multiple MCMC and optimization algorithms are also supported.

For an MGE source, the reduced number of free parameters (intensities are solved analytically, sigmas are
fixed, centres and ell_comps are shared) means that the sampler converges in fewer iterations than a
free-intensity Sersic source — even though each likelihood evaluation is slower per-iteration because of
the larger basis.

__Wrap Up__

We have presented a visual step-by-step guide to the multi-Gaussian expansion interferometer likelihood
function. The pipeline:

  ray-trace → source-plane Gaussian images (intensity=1 each) → `mapping_matrix` (N columns)
  → NUFFT → `transformed_mapping_matrix` → $D$ (length N) and $F$ (NxN) → solve $s = F^{-1} D$
  → `mapped_reconstructed_visibilities` → visibility-plane $\\chi^2$ → log likelihood

is the same as for the single-component linear light profile case
(`features/linear_light_profiles/likelihood_function.py`), just generalised to N basis components. The
NUFFT is the operation that nufftax made fast on the GPU, which is why this entire workflow is now
practical even for large MGE bases on visibility data.
"""
