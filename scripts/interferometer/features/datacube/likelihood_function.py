"""
__Log Likelihood Function: Datacube__

This script provides a step-by-step guide of the **PyAutoLens** `log_likelihood_function` used to fit a
**datacube** — a list of N per-channel `Interferometer` objects sharing a single lens model — with a per-channel
pixelized source reconstruction (specifically a `RectangularAdaptDensity` mesh and `Constant` regularization
scheme).

This script has the same aims as `interferometer/features/pixelization/likelihood_function.py`:

 - To provide a resource that authors can include in papers using **PyAutoLens**, so that readers can understand
 the likelihood function (including references to the previous literature from which it is defined) without
 having to write large quantities of text and equations.

 - To make inversions in **PyAutoLens** less of a "black-box" to users.

__Comparison To Single-Channel Pixelization Likelihood__

A datacube fit is identical to **N independent single-channel pixelization fits** plus one wrinkle: the lens
model is shared across all channels, so the same `tracer` is used for every channel's calculation. Inside any
single channel, every linear-algebra step matches `interferometer/features/pixelization/likelihood_function.py`
line for line:

 - Same ray-tracing (same lens galaxy mass + shear).
 - Same source-plane `Mapper` construction.
 - Same `mapping_matrix` (image-pixel to source-pixel mappings).
 - Same `transformed_mapping_matrix` (NUFFT applied per source pixel), with the *same* memory caveat as the
   single-channel case — for `n_vis ≳ 10^6` per channel the matrix becomes prohibitive to store and the
   sparse-operator likelihood function is the production path.
 - Same `data_vector`, `curvature_matrix`, `regularization_matrix`, NNLS reconstruction.
 - Same `chi_squared`, regularization term, complexity terms, noise normalisation, per-channel `log_evidence`.

The **cube log-evidence is the sum** of the per-channel log-evidences. That's it.

This script presents the calculation directly:

 1. Build the shared lens galaxy + source pixelization (channel-invariant).
 2. Walk through **channel 0** in detail (one short section per pixelization-script section, cross-referencing
    that script for the full derivations).
 3. Loop the per-channel calculation across all `dataset_list` and sum.
 4. Cross-check against per-channel `al.FitInterferometer.log_evidence`.

If you haven't read `interferometer/features/pixelization/likelihood_function.py` yet, do that first. This
script defers to it for almost everything that happens inside a single channel.

__Simplifications__

This example uses a `RectangularAdaptDensity` mesh + `Constant` regularization — the same combination used by
the rest of the `datacube/` tutorials (`modeling.py`, `start_here.py`). The
`pixelization/likelihood_function.py` reference uses `RectangularUniform`, which is a thin subclass of
`RectangularAdaptDensity`; the linear algebra is identical and the construction code is the same. The single
behaviour difference is that `RectangularAdaptDensity` lets the mesh's pixel density adapt to the source-plane
magnification map, which gives slightly better resolution in highly-magnified regions but does not change the
likelihood-function maths at all.

__Prerequisites__

The pixelization likelihood function is the most complex one in **PyAutoLens**. It is strongly advised you read
through the following first:

 - `interferometer/features/pixelization/likelihood_function.py` — full step-by-step walkthrough of the
   single-channel pixelization likelihood. This script is essentially a per-channel restatement of that one
   plus a sum, so the entire body below uses cross-references to its sections.
 - `interferometer/light_profile/log_likelihood_function.py` — the simpler light-profile likelihood, which
   introduces visibility-space inner products and the NUFFT without the pixelization linear algebra.

__Contents__

- **Comparison:** datacube = N independent pixelization fits + shared lens; cube log-evidence is the sum.
- **Simplifications:** `RectangularAdaptDensity` mesh, `Constant` regularization.
- **Prerequisites:** read `pixelization/likelihood_function.py` first.
- **Mesh Shape:** identical to the pixelization reference, sized to 14×14 to match `modeling.py`.
- **Mask:** identical to the pixelization reference, sized to the datacube simulator's 256×256 / 0.1″ grid.
- **Dataset:** load a *list* of `Interferometer` objects, one per channel (cube-specific).
- **Lens Galaxy:** identical to the pixelization reference (channel-invariant).
- **Source Galaxy Pixelization and Regularization:** identical to the pixelization reference.
- **One Channel Walkthrough:** run the full pixelization-likelihood calculation on channel 0, cross-referencing
  the single-channel script for shared derivations.
- **Across All Channels:** loop the per-channel calculation across `dataset_list` and sum — the headline
  cube-specific section.
- **Fit:** cross-check the manual sum against per-channel `al.FitInterferometer.log_evidence`.
- **Lens Modeling:** pointer to `modeling.py`, `start_here.py`, `delaunay.py`, `modeling_parametric.py`.
- **Log Likelihood Function: Source Code Speed Up:** per-channel fast paths apply, multiplied by N channels;
  the deferred shared-`Lᵀ W̃ L` optimisation recovers a factor-N speed-up when `uv_wavelengths`/`noise_map`
  are nearly channel-invariant.
- **Wrap Up:** pointers to modeling scripts and the planned JIT-correctness regression tests.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Mesh Shape__

Identical to `pixelization/likelihood_function.py:__Mesh Shape__`. Sized to 14×14 to match the
`datacube/modeling.py` mesh.
"""
mesh_pixels_yx = 14
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Mask__

Identical to `pixelization/likelihood_function.py:__Mask__`. Sized to match the datacube simulator's
256×256 / 0.1″ real-space grid.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256), pixel_scales=0.1, radius=3.0
)

"""
__Dataset__

This is the first genuinely cube-specific section. Instead of a single `al.Interferometer.from_fits(...)`
call, we load a *list* of `Interferometer` objects, one per channel.

The simulator at `simulator.py` writes the cube in two on-disk layouts:

 - **Per-channel folders** (used here): one `channel_NNN/` subfolder per channel with its own
   `data.fits`/`noise_map.fits`/`uv_wavelengths.fits`. Convenient when the channels are split-out already.
 - **3D-FITS cube** (`{visibilities,noise_map,uv_wavelengths}_cube.fits`, each `(n_chan, n_vis, 2)`): one
   file containing every channel stacked. `data_preparation.py` shows how to load this form. There is also a
   4D `(n_pol, n_chan, n_vis, 2)` CASA-like layout that requires a polarisation-collapse pre-processing step.

Whichever loader you use, you end up with the same list `dataset_list` below. Each entry has its own
`visibilities`, `noise_map`, and `uv_wavelengths`; the lens galaxy is channel-invariant and the per-channel
sources differ in `intensity` and `centre` (see `simulator.py`).
"""
dataset_label = "datacube"
dataset_name = "sim_simple"
dataset_path = Path("dataset") / "interferometer" / dataset_label / dataset_name

"""
__Dataset Auto-Simulation__

If the cube isn't on disk yet, run the simulator. This makes the script runnable on a fresh checkout.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/features/datacube/simulator.py"],
        check=True,
    )

channel_paths = sorted(
    p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("channel_")
)

dataset_list = [
    al.Interferometer.from_fits(
        data_path=channel_path / "data.fits",
        noise_map_path=channel_path / "noise_map.fits",
        uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    for channel_path in channel_paths
]

n_channels = len(dataset_list)
n_vis = int(dataset_list[0].uv_wavelengths.shape[0])
print(f"Loaded {n_channels} channels, {n_vis} visibilities per channel.")

"""
We can plot the dirty images of the first and last channels to see the per-channel data the cube fit will
work with. The source intensity and centre vary across the cube, so the dirty images differ even though the
lens model is the same.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset_list[0])
aplt.subplot_interferometer_dirty_images(dataset=dataset_list[-1])

"""
__Over Sampling__

Same as `pixelization/likelihood_function.py:__Over Sampling__`. Interferometer pixelizations do not use
over-sampling.

__Masked Image Grid__

Identical to `pixelization/likelihood_function.py:__Masked Image Grid__`. Every channel uses the same
real-space mask, so `dataset.grids.pixelization` is channel-invariant.
"""
aplt.plot_grid(grid=dataset_list[0].grids.pixelization, title="")

"""
__Lens Galaxy__

Identical to `pixelization/likelihood_function.py:__Lens Galaxy__`. The lens galaxy is channel-invariant
(this is the entire reason datacube modeling can share a single non-linear search across all channels — the
mass model doesn't depend on frequency).
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)
shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
lens_galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)

"""
__Source Galaxy Pixelization and Regularization__

Same as `pixelization/likelihood_function.py:__Source Galaxy Pixelization and Regularization__`, with
`RectangularAdaptDensity` substituted for `RectangularUniform`. The classes share the same construction
machinery — `RectangularUniform` is a subclass of `RectangularAdaptDensity` — so all of the mesh-grid and
mapper code below is unchanged.

The same source pixelization is used for every channel. Each channel runs its own linear inversion against
this shared pixelization (which is what gives each channel an independent source-plane reconstruction).
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)
source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Ray Tracing__

Identical to `pixelization/likelihood_function.py:__Ray Tracing__`. Channel-invariant because the lens is
channel-invariant; the same `tracer` is used by every per-channel calculation below.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])


"""
==================================================================================================
                                  ONE CHANNEL WALKTHROUGH
==================================================================================================

We now walk through the full pixelization-likelihood calculation for **channel 0**. Every section heading
below matches a section in `pixelization/likelihood_function.py` and the calculation is the same. Each
section has a one-line cross-reference, the code (so this script runs end-to-end), and any cube-specific
notes that don't apply in the single-channel case.
"""

dataset = dataset_list[0]
print(f"\n=== Channel 0 walkthrough ===")

"""
__Border Relocation__

Identical to `pixelization/likelihood_function.py:__Border Relocation__`. Run per channel because the traced
grid feeding into it depends on the per-channel `dataset.grids.pixelization` — but `dataset.grids.pixelization`
is itself channel-invariant in this cube (every channel uses the same `real_space_mask`), so the relocated
grid actually only needs to be computed once. We compute it inside the loop below for clarity.
"""
from autoarray.inversion.mesh.border_relocator import BorderRelocator

traced_grid_pixelization = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization
)[-1]

border_relocator = BorderRelocator(mask=dataset.mask, sub_size=1)
relocated_grid = border_relocator.relocated_grid_from(grid=traced_grid_pixelization)

aplt.plot_grid(grid=relocated_grid, title="Channel 0 relocated traced grid")

"""
__Source Pixel Centre Calculation__

Identical to `pixelization/likelihood_function.py:__Source Pixel Centre Calculation__`. The source-plane mesh
overlays the relocated traced grid; since the relocated grid is the same for every channel (shared mask,
shared tracer), the source-plane mesh grid is channel-invariant too.
"""
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import overlay_grid_from

mesh_grid = overlay_grid_from(
    shape_native=mesh_shape, grid=al.Grid2DIrregular(relocated_grid)
)

"""
__Interpolation / Mapper / Mapping Matrix__

Identical to `pixelization/likelihood_function.py:__Interpolation__`, `__Mapper__`, and
`__Mapping Matrix__`. All channel-invariant for the cube — they depend only on the shared `tracer`,
mesh shape and mask. We compute them once and reuse across channels in the loop below.
"""
interpolator = pixelization.mesh.interpolator_from(
    source_plane_data_grid=relocated_grid,
    source_plane_mesh_grid=mesh_grid,
)
mapper = al.Mapper(interpolator=interpolator)

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.over_sampler.sub_fraction,
)

print(f"  mapping_matrix shape: {mapping_matrix.shape} (real-space pixels x source pixels)")

"""
__Transformed Mapping Matrix ($f$)__

For a single channel this is identical to `pixelization/likelihood_function.py:__Transformed Mapping Matrix__`.

**This is where per-channel structure starts to appear.** Each channel has its own `uv_wavelengths` and
therefore its own NUFFT operator (`dataset.transformer`). The shape of the output is the same for every
channel in our simulator (every channel has the same `n_vis`), but the values differ — the same source pixel
maps to different uv-plane visibilities in different channels because the baselines differ.

The same memory caveat applies *per channel*: for `n_vis ≳ 10^6` per channel, this matrix is many GB per
channel; for an N-channel cube the total memory footprint is N× the single-channel one. The sparse-operator
likelihood function — see the `apply_sparse_operator` calls in `modeling.py` — is the production path that
avoids ever materialising this matrix. The deferred shared-`Lᵀ W̃ L` optimisation goes further: it computes
the curvature-matrix sandwich once and reuses it across channels when `uv_wavelengths`/`noise_map` are
nearly channel-invariant (which they typically are for narrow emission lines).
"""
transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
    mapping_matrix=mapping_matrix
)

print(
    f"  transformed_mapping_matrix shape: {transformed_mapping_matrix.shape} "
    f"(n_vis x source pixels), dtype: {transformed_mapping_matrix.dtype}"
)

"""
__Data Vector (D)__

Identical to `pixelization/likelihood_function.py:__Data Vector (D)__`. Per channel because each channel has
its own `dataset.data` (visibilities) and `dataset.noise_map`; the construction formula is unchanged.
"""
data_vector = (
    al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=dataset.data,
        noise_map=dataset.noise_map,
    )
)

"""
__Curvature Matrix (F)__

Identical to `pixelization/likelihood_function.py:__Curvature Matrix (F)__`. The matrix `F` depends only on
`transformed_mapping_matrix` and `noise_map` — both of which are channel-specific because of the per-channel
NUFFT and noise — but for the typical narrow-emission-line case where `uv_wavelengths` and `noise_map` change
very little across the line, `F` is nearly channel-invariant. This is the matrix Aris's deferred
shared-`Lᵀ W̃ L` optimisation reuses across channels.
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

"""
__Regularization Matrix (H)__

Identical to `pixelization/likelihood_function.py:__Regularization Matrix (H)__`. Channel-invariant because
the regularization scheme + mesh structure are channel-invariant — `H` only depends on which source pixels
neighbour which.
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.neighbors,
    neighbors_sizes=mapper.neighbors.sizes,
)

"""
__F + λH / Galaxy Reconstruction (s)__

Identical to `pixelization/likelihood_function.py:__F + Lamdba H__` and `__Galaxy Reconstruction (s)__`. The
linear system `s = (F + λH)⁻¹ D` is solved per channel, producing each channel's source-plane reconstruction.

Per-channel reconstructions are what make the cube fit physically interesting: an emission line that
brightens-and-fades across the cube produces a sequence of source-plane reconstructions whose total flux
traces the line profile, while the lens mass model stays fixed.
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

"""
__Visibilities Reconstruction__

Identical to `pixelization/likelihood_function.py:__Visibilities Reconstruction__`. Per channel because the
model visibilities live in the channel's own uv-plane.
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

"""
__Likelihood Function — Five Terms__

Identical to `pixelization/likelihood_function.py:__Likelihood Function__`. The same five-term formula
applies per channel:

  -2 ln ε_c = χ²_c + s_cᵀ H s_c + ln det(F_c + H) - ln det(H) + Σ_j ln (2π σ²_{c,j})

where `c` indexes channels. The cube log-evidence is `Σ_c log_evidence_c`, computed in the
"Across All Channels" section below.

__Chi Squared__

Identical to `pixelization/likelihood_function.py:__Chi Squared__`. Per channel — each channel has its own
visibilities, noise, and model.
"""
model_visibilities = mapped_reconstructed_visibilities
residual_map = dataset.data - model_visibilities

chi_squared_map_real = (residual_map.real / dataset.noise_map.real) ** 2
chi_squared_map_imag = (residual_map.imag / dataset.noise_map.imag) ** 2
chi_squared = np.sum(chi_squared_map_real) + np.sum(chi_squared_map_imag)

"""
__Regularization Term__

Identical to `pixelization/likelihood_function.py:__Regularization Term__`. Per channel via the per-channel
`reconstruction`; `H` is channel-invariant.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

"""
__Complexity Terms__

Identical to `pixelization/likelihood_function.py:__Complexity Terms__`. The `ln det(F + λH)` term is
per-channel (`F` is per-channel); `ln det(H)` is channel-invariant.
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

"""
__Noise Normalisation Term__

Identical to `pixelization/likelihood_function.py:__Noise Normalization Term__`. Per channel.

**Cube-specific note**: if you did polarisation-collapse by *averaging* (see `data_preparation.py`), the
visibility noise map already incorporates the sqrt(2) noise reduction relative to a single polarisation. The
noise-normalisation term you compute below matches whatever your data-prep produced — it doesn't double-count
the polarisation averaging. If you concatenated polarisations instead, `n_vis` doubled and the term naturally
scales with it.
"""
noise_normalization_real = np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
noise_normalization_imag = np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
noise_normalization = noise_normalization_real + noise_normalization_imag

"""
__Calculate The Log Likelihood (Channel 0)__

Identical to `pixelization/likelihood_function.py:__Calculate The Log Likelihood__`. The result is the
log-evidence of channel 0 only.
"""
log_evidence_channel_0 = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)
print(f"  channel 0 log_evidence: {log_evidence_channel_0:.6f}")


"""
==================================================================================================
                                    ACROSS ALL CHANNELS
==================================================================================================

This is the headline cube-specific section. We loop the per-channel calculation across `dataset_list` and
sum to get the cube log-evidence.

Almost everything inside the loop is channel-invariant — the `tracer`, `mapper`, `mapping_matrix`, and
`regularization_matrix`. Only the per-channel `dataset` (visibilities, noise_map, uv_wavelengths) changes, so
inside the loop we only redo the channel-dependent steps:

 - `transformed_mapping_matrix` (depends on `dataset.transformer`).
 - `data_vector` (depends on `dataset.data` and `dataset.noise_map`).
 - `curvature_matrix` (depends on `transformed_mapping_matrix` and `dataset.noise_map`).
 - `reconstruction` (depends on `data_vector` and `curvature_matrix`).
 - Visibilities-space residuals, χ², regularization term, complexity terms, noise normalisation.

This is exactly what `af.FactorGraphModel` does internally for the modeling scripts: it feeds the same
lens-model parameters to every per-channel `AnalysisInterferometer.log_likelihood_function`, and the search
sees the sum.
"""


def per_channel_log_evidence(dataset):
    """Compute the log-evidence of a single channel given the channel-invariant `tracer`, `mapper`,
    `mapping_matrix` and `regularization_matrix` defined above.

    All steps mirror `pixelization/likelihood_function.py` line by line; only the dataset (visibilities,
    noise map, uv_wavelengths) varies.
    """

    transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    data_vector = (
        al.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrix,
            visibilities=dataset.data,
            noise_map=dataset.noise_map,
        )
    )

    real_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrix.real,
        noise_map=dataset.noise_map.real,
    )
    imag_curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrix.imag,
        noise_map=dataset.noise_map.imag,
    )
    curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)
    curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

    reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

    mapped_reconstructed_visibilities = (
        al.util.inversion_interferometer.mapped_reconstructed_visibilities_from(
            transformed_mapping_matrix=transformed_mapping_matrix,
            reconstruction=reconstruction,
        )
    )
    model_visibilities = al.Visibilities(
        visibilities=mapped_reconstructed_visibilities
    )

    residual_map = dataset.data - model_visibilities
    chi_squared = float(
        np.sum((residual_map.real / dataset.noise_map.real) ** 2)
        + np.sum((residual_map.imag / dataset.noise_map.imag) ** 2)
    )

    regularization_term = float(
        np.matmul(reconstruction.T, np.matmul(regularization_matrix, reconstruction))
    )
    log_curvature_reg_matrix_term = float(np.linalg.slogdet(curvature_reg_matrix)[1])
    log_regularization_matrix_term = float(np.linalg.slogdet(regularization_matrix)[1])

    noise_normalization = float(
        np.sum(np.log(2 * np.pi * dataset.noise_map.real**2.0))
        + np.sum(np.log(2 * np.pi * dataset.noise_map.imag**2.0))
    )

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )


print(f"\n=== Across all channels ===")
per_channel_log_evidences = [per_channel_log_evidence(d) for d in dataset_list]
for c, le in enumerate(per_channel_log_evidences):
    print(f"  channel {c}: log_evidence = {le:.6f}")

cube_log_evidence = sum(per_channel_log_evidences)
print(f"  cube log_evidence = sum(per_channel) = {cube_log_evidence:.6f}")


"""
__Fit__

The whole per-channel block above is wrapped inside `al.FitInterferometer` — exactly as in the single-channel
case at `pixelization/likelihood_function.py:__Fit__`. We loop the `FitInterferometer` construction across
`dataset_list` and print the summed `fit.log_evidence` alongside our manual computation.

The two values will agree to ~3 significant figures but typically not exactly. The small residual difference
(~0.05% relative) comes from source-code internals that this walkthrough deliberately doesn't reproduce — the
`__Log Likelihood Function: Source Code Speed Up__` section below describes the production-fast versions of
`chi_squared` and `curvature_matrix` that bypass the dense `transformed_mapping_matrix`. The pixelization
reference exhibits the same discrepancy.
"""
fits = []
print(f"\n=== Cross-check vs FitInterferometer ===")
for c, dataset in enumerate(dataset_list):
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
    )
    fits.append(fit)
    print(f"  channel {c}: FitInterferometer.log_evidence = {fit.log_evidence:.6f}")

fit_total_log_evidence = sum(fit.log_evidence for fit in fits)
print(f"  summed FitInterferometer.log_evidence = {fit_total_log_evidence:.6f}")
print(f"  manual cube_log_evidence              = {cube_log_evidence:.6f}")
print(
    f"  relative difference                   = "
    f"{abs(cube_log_evidence - fit_total_log_evidence) / abs(fit_total_log_evidence):.2e}"
)

"""
__Lens Modeling__

To fit a lens model to a datacube, this likelihood function is sampled across many candidate lens-model
parameters using a non-linear search. For the user-facing modeling story see:

 - `modeling.py` — `RectangularAdaptDensity` pixelization fit with `af.Nautilus`, the canonical entry point.
 - `start_here.py` — narrative walkthrough wrapping the same fit.
 - `delaunay.py` — Delaunay-pixelized source variant.
 - `modeling_parametric.py` — parametric `Sersic` source variant (per-channel intensity).

All four use `af.FactorGraphModel` to wrap a list of `AnalysisInterferometer` objects — the framework's way
of expressing the explicit cube sum we just walked through. Internally the FactorGraph routes the same
lens-model parameters to every per-channel `AnalysisInterferometer.log_likelihood_function` and sums.

__Log Likelihood Function: Source Code Speed Up__

The pixelization-likelihood guide's `__Log Likelihood Function: Source Code Speed Up__` section applies
unchanged per channel:

 - **Fast chi-squared:** the source code never materialises `transformed_mapping_matrix` to compute χ².
 - **Sparse-operator curvature matrix:** likewise for `F`.

For an N-channel cube these speed-ups apply per channel, multiplied by N. The deferred shared-`Lᵀ W̃ L`
optimisation Aris designed goes further: when `uv_wavelengths` and `noise_map` are nearly channel-invariant
(the typical narrow-emission-line case), the curvature-matrix sandwich `Lᵀ W̃ L` can be computed once and
reused across all N channels. That recovers a factor-N speed-up on top of the per-channel sparse-operator
gains, which is what brings ALMA-scale cubes back inside CPU runtime budgets.

__Wrap Up__

This script presented the cube likelihood function as **N independent pixelization likelihoods** + **a shared
lens model** + **a sum**.

For deeper dives:

 - `interferometer/features/pixelization/likelihood_function.py` — the single-channel pixelization
   walkthrough this script defers to for the per-channel internals.
 - `modeling.py` / `start_here.py` / `delaunay.py` / `modeling_parametric.py` — user-facing modeling scripts
   that wrap this likelihood in `af.FactorGraphModel` + `af.Nautilus`.
 - `data_preparation.py` — how to bridge from CASA's 4D `(n_pol, n_chan, n_vis, 2)` output to the per-channel
   `Interferometer` objects this walkthrough loads.

A planned `autolens_workspace_test/scripts/jax_likelihood_functions/datacube/` folder will hold end-to-end
JAX-JIT correctness tests for the cube likelihood. The JIT-vs-eager `rtol=1e-4` regression that previously
lived in this file moves there once those test scripts land.
"""
