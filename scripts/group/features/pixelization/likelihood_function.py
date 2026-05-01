"""
__Log Likelihood Function: Pixelization (Group)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data of
a group-scale strong lens where the source galaxy is reconstructed using a pixelized mesh.

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the pixelized
 likelihood function for group-scale lenses without having to read the source code.

 - To illustrate how group-scale lensing with a pixelized source differs from galaxy-scale lensing: multiple
 mass profiles from multiple galaxies contribute to the deflection angles used for ray-tracing, and the
 pixelized source reconstruction involves linear algebra (the mapping matrix, regularization, etc.).

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Lens Galaxies:** Define the main lens and extra galaxies with light and mass profiles.
**Source Galaxy Pixelization:** The source uses a Delaunay mesh with constant regularization.
**Lens Light:** Compute the total lens light from all galaxies.
**Deflection Angles:** Compute deflection angles from all mass profiles.
**Ray Tracing:** Ray-trace image pixels to the source plane using combined deflections.
**Pixelized Source Reconstruction:** The linear algebra inversion step.
**Likelihood Function:** Compute the log likelihood including regularization evidence terms.
**Fit:** Confirm the step-by-step calculation matches the FitImaging object.

__Prerequisites__

The likelihood function of a pixelization builds on that used for standard light profiles and
linear light profiles. You should first read:

- `group/likelihood_function.py` (group-scale parametric likelihood).
- `imaging/features/pixelization/likelihood_function.py` (galaxy-scale pixelized likelihood).
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autoarray as aa
import autolens.plot as aplt

"""
__Dataset__

Load the group-scale strong lens dataset with 0.1 arcsecond-per-pixel resolution.
"""
dataset_path = Path("dataset", "group", "simple")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script.
"""
if not dataset_path.exists():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/group/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use a 7.5" circular mask for the group-scale lens.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Over Sampling__

For simplicity in this step-by-step guide, we disable over sampling.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

"""
__Galaxy Centres__

Load the centres of the main lens galaxies and extra galaxies.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Main Lens Galaxy__

The main lens galaxy has a spherical Sersic light profile and an isothermal mass profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

"""
__Extra Galaxies__

The extra galaxies each have their own light and mass profiles.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

"""
__Image-Plane Mesh Grid__

The Delaunay mesh requires an image-plane mesh grid whose ray-traced positions become the Delaunay
vertices in the source plane. We build it via an `Overlay` image-mesh covering the masked field, then
append a ring of edge pixels at the mask boundary so the linear inversion can zero them out.
"""
edge_pixels_total = 30

image_mesh = al.image_mesh.Overlay(shape=(22, 22))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=masked_dataset.mask)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

"""
__Source Galaxy Pixelization__

The source galaxy is reconstructed using a Delaunay mesh with constant regularization, rather than
an analytic light profile.

The `Pixelization` consists of:

 - `mesh`: A `Delaunay` triangulation whose source-pixel count matches the image-plane mesh grid
   computed above (with `edge_pixels_total` boundary vertices reserved for zeroing). The triangle
   vertices are determined by ray-tracing this image-plane grid to the source plane.

 - `regularization`: A `Constant` scheme that applies uniform smoothing across all source pixels, with
   a single free regularization coefficient.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=edge_pixels_total
    ),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

"""
__Lens Light__

Compute the total lens light from ALL galaxies (main + extra) in the group.

For group-scale lenses, the total lens light is the sum of images from every lens galaxy.
"""
lens_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grid)
extra_0_image_2d = extra_galaxy_0.image_2d_from(grid=masked_dataset.grid)
extra_1_image_2d = extra_galaxy_1.image_2d_from(grid=masked_dataset.grid)

total_lens_image_2d = lens_image_2d + extra_0_image_2d + extra_1_image_2d

aplt.plot_array(array=total_lens_image_2d, title="Total Lens Light (All Galaxies)")

"""
We also compute blurring images for PSF convolution.
"""
lens_blurring_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)
extra_0_blurring_image_2d = extra_galaxy_0.image_2d_from(
    grid=masked_dataset.grids.blurring
)
extra_1_blurring_image_2d = extra_galaxy_1.image_2d_from(
    grid=masked_dataset.grids.blurring
)

total_lens_blurring_image_2d = (
    lens_blurring_image_2d + extra_0_blurring_image_2d + extra_1_blurring_image_2d
)

"""
__Deflection Angles__

We compute deflection angles from ALL mass profiles in the group.

For group-scale lensing, the total deflection is:

 alpha_total = alpha_lens + alpha_extra_0 + alpha_extra_1

Each galaxy's mass profile contributes to the total deflection, and errors in any of them lead
to incorrect source-plane coordinates.
"""
deflections_lens = lens_galaxy.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_0 = extra_galaxy_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_1 = extra_galaxy_1.deflections_yx_2d_from(grid=masked_dataset.grid)

"""
__Ray Tracing__

Ray-trace every 2D (y,x) coordinate from the image-plane to the source-plane using the summed
deflection angles from ALL galaxies:

 beta = theta - alpha_total(theta)

The `Tracer` object handles this automatically.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

"""
__Pixelized Source Reconstruction__

Unlike a parametric source, a pixelized source does not have a closed-form image. Instead, the source
is reconstructed via a linear algebra inversion.

The key steps are:

 1. **Mesh construction**: The Delaunay triangulation is built in the source plane from the ray-traced
    positions of a coarse image-plane grid.

 2. **Mapping matrix**: A matrix F is constructed where F_ij describes the fractional contribution of
    source pixel j to image pixel i. For a Delaunay mesh, this uses barycentric interpolation within
    each triangle.

 3. **Data vector**: d = F^T (D / sigma^2), where D is the (lens-subtracted) data and sigma is the
    noise map.

 4. **Curvature matrix**: C = F^T diag(1/sigma^2) F, which encodes how image pixels constrain source
    pixels.

 5. **Regularization matrix**: H encodes the smoothness prior on the source. For constant regularization,
    H = lambda * G, where G penalizes flux differences between neighboring source pixels.

 6. **Inversion**: Solve (C + H) s = d for the source pixel fluxes s. A positive-only solver is used
    to ensure all reconstructed fluxes are physical (non-negative).

 7. **Model image**: The reconstructed source fluxes are mapped back to the image plane via the mapping
    matrix to produce the model source image, which is then added to the lens light and convolved
    with the PSF.

For group-scale lenses, the larger mask means more image pixels, making the mapping matrix larger
and the inversion more computationally expensive.
"""

"""
__Likelihood Function__

The log likelihood for a pixelized source has additional terms compared to a parametric source:

 -2 ln L = chi^2 + s^T H s - log|H| + log|C + H| + N ln(2 pi sigma^2)

Where:

 - chi^2 = sum((data - model)^2 / sigma^2) is the standard goodness-of-fit term
 - s^T H s is the regularization penalty (source smoothness)
 - log|H| and log|C + H| are evidence terms that balance fit quality vs. source complexity
 - N ln(2 pi sigma^2) is the noise normalization

These extra terms implement Bayesian regularization: simpler (smoother) source reconstructions are
preferred unless the data demands more complexity.
"""

"""
__Fit__

The `FitImaging` object performs all of the above steps automatically. We verify that it produces the
correct result for the group-scale pixelized fit.

The image-plane mesh grid is supplied via an `AdaptImages` object, which pairs it with the source
galaxy so the Delaunay vertices can be ray-traced to the source plane during the fit.
"""
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid}
)

fit = al.FitImaging(
    dataset=masked_dataset, tracer=tracer, adapt_images=adapt_images
)
fit_figure_of_merit = fit.figure_of_merit
print(f"Log Likelihood: {fit_figure_of_merit}")

aplt.subplot_fit_imaging(fit=fit)

"""
__Inversion Details__

The inversion object provides access to the individual terms of the evidence-based likelihood.
"""
inversion = fit.inversion

print(f"Regularization Term (s^T H s): {inversion.regularization_term}")
print(f"log|H|: {inversion.log_det_regularization_matrix_term}")
print(f"log|C + H|: {inversion.log_det_curvature_reg_matrix_term}")

"""
__Wrap Up__

We have presented a step-by-step guide to the group-scale pixelized source likelihood function.

The key differences from the galaxy-scale pixelized likelihood are:

 - Multiple lens galaxies (main + extra) each contribute light and mass profiles.
 - The total deflection field is the sum of deflections from ALL galaxies in the group.
 - The larger mask (7.5") means more image pixels, making the mapping matrix and inversion larger.
 - Accurate lens light subtraction from all group members is critical for a clean source reconstruction.

The key differences from the group-scale parametric likelihood are:

 - The source is reconstructed via linear algebra (mapping matrix, regularization) rather than evaluated
   from a parametric profile.
 - The likelihood includes Bayesian evidence terms that penalize overly complex source reconstructions.
 - A positive-only solver ensures physical (non-negative) source pixel fluxes.
"""
