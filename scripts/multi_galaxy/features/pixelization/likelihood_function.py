"""
__Log Likelihood Function: Pixelization (Multi Galaxy)__

This script walks through the log likelihood function of a multi-galaxy lens fitted with a **pixelized source**,
step by step, so the calculation can be followed without reading the source code.

__Prerequisites__

Read these first; they are not repeated here:

 - `multi_galaxy/likelihood_function.py` — the same walkthrough with a parametric source. It covers the summed
   deflection field and the multi-deflector light subtraction in full.
 - `imaging/features/pixelization/likelihood_function.py` — the galaxy-scale pixelized derivation, which
   develops the mapping matrix, the data vector, the curvature matrix and the evidence terms in detail.

__Contents__

- **What Changes:** Which steps a second deflector touches, and which it does not.
- **Dataset, Mask & Over Sampling:** Set up.
- **Main Lens Galaxies:** The two co-dominant deflectors.
- **Source Galaxy Pixelization:** A rectangular mesh with constant regularization.
- **Lens Light:** The summed light of both deflectors, which is subtracted before the inversion.
- **Deflection Angles:** One deflection field per deflector.
- **Ray Tracing:** Tracing the image-plane grid with the summed deflections.
- **Pixelized Source Reconstruction:** The inversion, step by step.
- **Likelihood Function:** The evidence terms a pixelization adds.
- **Fit:** Confirm the walkthrough matches `FitImaging`.
- **Inversion Details:** Read the individual likelihood terms off the inversion.
- **Wrap Up:** Where to go next.

__What Changes__

A pixelized likelihood evaluation has two halves, and a second co-dominant deflector reaches only one of them:

 - **Before the inversion** — the lens light subtracted from the data is the sum of both deflectors' light, and
   the deflection field used to ray-trace is the sum of both deflectors' mass. Both are sums over galaxies
   rather than single-galaxy quantities.

 - **The inversion itself** — the mapping matrix, the regularization matrix, the linear solve and the evidence
   terms are computed from the traced grid and the source mesh. They see only the traced coordinates, so their
   form is identical to the galaxy-scale case.

The deflectors reach the inversion only through the grid they trace, which is why this script spends its length
on the first half and defers the second to the galaxy-scale walkthrough.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np

import autolens as al
import autolens.plot as aplt

"""
__Dataset__

The `simple` multi-galaxy dataset, the same co-dominant pair fitted by `multi_galaxy/modeling.py`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
dataset_name = "simple"
dataset_path = Path("dataset", "multi_galaxy", dataset_name)

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_galaxy/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.05,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Extra Galaxies Noise Scaling__

Scale the faint contaminant out of the fit, as `multi_galaxy/modeling.py` explains. This happens before anything
below, so the contaminant's pixels carry no weight in the data vector the inversion builds.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

"""
__Mask__

The standard 3.0" circular mask.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

Over sampling is disabled for this step-by-step guide, so each image pixel is one coordinate and the arrays
below are easy to follow. A real fit over-samples at every deflector centre, as
`multi_galaxy/features/pixelization/modeling.py` does.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

"""
__Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

print(f"Deflector centres: {main_lens_centres}")

"""
__Main Lens Galaxies__

The two co-dominant deflectors, at the values used by `multi_galaxy/simulator.py`. Their light is a plain
`Sersic` with an input intensity rather than a linear profile, so the light subtraction below is a straight
subtraction and the only linear algebra in this script is the source's.
"""
lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.35, 0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.2,
        effective_radius=0.6,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.30, 0.28),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
        einstein_radius=1.0,
    ),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.35, -0.25),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.31, -0.22),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

shear_galaxy = al.Galaxy(
    redshift=0.5,
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

"""
__Source Galaxy Pixelization__

The source has no light profile. Instead it carries a `Pixelization`, built from two objects:

 - `mesh`: a `RectangularUniform` grid of source pixels, whose shape is fixed before the fit because it sets the
   size of every matrix below.

 - `regularization`: a `Constant` scheme, which penalizes flux differences between neighbouring source pixels
   with a single coefficient.

`multi_galaxy/features/pixelization/delaunay.py` uses a Delaunay mesh instead, where the source pixels are
triangles whose vertices are ray-traced from the image plane.
"""
mesh_shape = (30, 30)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Lens Light__

The lens light subtracted from the data is the sum of both deflectors' images. Each galaxy's image is computed
on the same grid and added.
"""
lens_0_image_2d = lens_0.image_2d_from(grid=masked_dataset.grid)
lens_1_image_2d = lens_1.image_2d_from(grid=masked_dataset.grid)

total_lens_image_2d = lens_0_image_2d + lens_1_image_2d

aplt.plot_array(array=total_lens_image_2d, title="Total Lens Light (Both Deflectors)")

"""
The blurring grid holds the pixels just outside the mask whose light is convolved into it by the PSF. Both
deflectors contribute to it, and both contributions are summed the same way.
"""
lens_0_blurring_image_2d = lens_0.image_2d_from(grid=masked_dataset.grids.blurring)
lens_1_blurring_image_2d = lens_1.image_2d_from(grid=masked_dataset.grids.blurring)

total_lens_blurring_image_2d = lens_0_blurring_image_2d + lens_1_blurring_image_2d

"""
__Deflection Angles__

Each deflector's mass profile produces its own deflection field, and the shear galaxy produces a third. The
total deflection at each image coordinate is their sum:

 alpha_total = alpha_lens_0 + alpha_lens_1 + alpha_shear

This is the step where a second co-dominant deflector enters the likelihood, and it is the only step where the
mass models are used.
"""
deflections_lens_0 = lens_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_lens_1 = lens_1.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_shear = shear_galaxy.deflections_yx_2d_from(grid=masked_dataset.grid)

deflections_total = deflections_lens_0 + deflections_lens_1 + deflections_shear

"""
__Ray Tracing__

Every image-plane coordinate is traced to the source plane by subtracting the total deflection:

 beta = theta - alpha_total(theta)

The `Tracer` does this for us. Both deflectors are at the same redshift, so this is a single-plane trace and the
summed deflection field above is exactly what the tracer uses.
"""
tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

"""
The traced grid is the only thing the inversion receives from the lens model. Two deflectors, one deflector or
ten, the steps below are identical once this grid exists.
"""
print(f"Number of traced image coordinates: {traced_grid.shape[0]}")

"""
__Pixelized Source Reconstruction__

A pixelized source has no closed-form image. Its pixel fluxes are solved by a linear inversion, whose steps are:

 1. **Mesh construction:** the rectangular mesh is laid over the extent of the traced grid, so its source pixels
    cover where the image pixels actually land.

 2. **Mapping matrix:** a matrix `f` whose entry `f_ij` is the fraction of image pixel `i` that comes from
    source pixel `j`. It is then convolved with the PSF to give the blurred mapping matrix.

 3. **Data vector:** `D = f^T (d / sigma^2)`, where `d` is the lens-light-subtracted data and `sigma` the noise
    map.

 4. **Curvature matrix:** `F = f^T diag(1 / sigma^2) f`, which encodes how strongly the image pixels constrain
    each source pixel and how source pixels are correlated with one another.

 5. **Regularization matrix:** `H`, the smoothness prior. For `Constant` regularization this is one coefficient
    times a matrix penalizing flux differences between neighbouring source pixels.

 6. **Inversion:** solve `(F + H) s = D` for the source fluxes `s`, with a positive-only solver so no source
    pixel reconstructs negative flux.

 7. **Model image:** map `s` back through the mapping matrix, add the lens light, and convolve with the PSF.

None of these steps counts the number of deflectors. They act on the traced grid produced above.

__Likelihood Function__

A pixelized source adds evidence terms to the chi-squared:

 -2 ln L = chi^2 + s^T H s - ln|H| + ln|F + H| + N ln(2 pi sigma^2)

where:

 - `chi^2 = sum((d - model)^2 / sigma^2)` is the goodness of fit.
 - `s^T H s` penalizes an unsmooth reconstruction.
 - `ln|H|` and `ln|F + H|` are the Bayesian evidence terms, which trade fit quality against source complexity.
 - `N ln(2 pi sigma^2)` is the noise normalization.

The evidence terms are what stop the source absorbing whatever the mass model leaves behind: a reconstruction
that needs more structure to fit the data pays for it here.

__Fit__

`FitImaging` performs every step above. Running it confirms the walkthrough describes the same calculation.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

print(f"Log Likelihood: {fit.log_likelihood}")

aplt.subplot_fit_imaging(fit=fit)

"""
__Inversion Details__

The individual terms of the expression above are available on the inversion, which is useful when checking which
term a change to the model actually moved.
"""
inversion = fit.inversion

print(f"Number of source pixels: {inversion.reconstruction.shape[0]}")
print(f"Regularization term (s^T H s): {inversion.regularization_term}")
print(f"ln|H|: {inversion.log_det_regularization_matrix_term}")
print(f"ln|F + H|: {inversion.log_det_curvature_reg_matrix_term}")

"""
The reconstruction itself is a flat array of fluxes, one per source pixel, in the mesh's own ordering.
"""
print(f"Total reconstructed source flux: {np.sum(inversion.reconstruction)} e- s^-1")

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/features/pixelization/fit.py` — the same fit without the step-by-step breakdown.
 - `multi_galaxy/features/pixelization/modeling.py` — fitting this model with a non-linear search.
 - `multi_galaxy/likelihood_function.py` — the parametric-source walkthrough, which covers the summed
   deflection field and light subtraction in more depth.
 - `imaging/features/pixelization/likelihood_function.py` — the galaxy-scale derivation of every inversion step
   listed above, including the mapping matrix and evidence terms in full.
"""
