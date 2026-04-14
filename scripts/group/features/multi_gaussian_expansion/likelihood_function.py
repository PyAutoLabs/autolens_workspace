"""
__Log Likelihood Function: Multi Gaussian Expansion (Group)__

This script provides a step-by-step guide of the `log_likelihood_function` which is used to fit `Imaging` data of
a group-scale strong lens, with a focus on how Multi Gaussian Expansion (MGE) light profiles fit into the
likelihood calculation.

An MGE decomposes each galaxy's light into ~10-30+ Gaussians whose intensities are solved via linear algebra.
For group-scale lenses, this means multiple galaxies can each have their own set of Gaussians, and the total
model image is the sum of all Gaussian images from all galaxies, convolved with the PSF.

This script uses simple `SersicSph` light profiles for the concrete step-by-step calculation (as specifying
concrete MGE instances requires many parameters), but explains how the MGE likelihood differs at each step.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Over Sampling:** Disable over sampling for simplicity.
**Main Lens Galaxy:** The main lens galaxy at the centre of the group.
**Extra Galaxies:** The two extra galaxies are companion galaxies near the main lens.
**Source Galaxy:** The source galaxy light profile.
**Lens Light:** Compute a 2D image of each lens galaxy's light and sum them together.
**Lens Galaxy Mass:** Compute deflection angles from all mass profiles.
**Ray Tracing:** Ray-trace the image-plane grid to the source plane.
**Source Image:** Evaluate the source galaxy light on the traced grid.
**Convolution:** Convolve the total image with the PSF.
**Likelihood Function:** Compute the log likelihood.
**MGE Likelihood:** How the likelihood changes when using MGE light profiles.
**Fit:** Verify using the FitImaging object.

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

For simplicity, we disable over sampling in this guide by setting `sub_size=1`.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Main Lens Galaxy__

The main lens galaxy is at the centre of the group. It has a spherical Sersic light profile and a spherical
isothermal mass profile.

With an MGE, this galaxy's light would instead be decomposed into ~20 Gaussians, each with a different sigma
value spanning from 0.01" to the mask radius. The intensities of these Gaussians are solved via linear algebra
rather than being specified manually.
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

The two extra galaxies are companion galaxies near the main lens. With an MGE, each extra galaxy's light
would be decomposed into ~10 Gaussians with centres fixed to the observed positions. Crucially, this adds
zero non-linear parameters per extra galaxy, unlike a Sersic profile which adds 5.
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
__Source Galaxy Light Profile__

The source galaxy uses a cored elliptical Sersic. With an MGE source model, this would instead be ~20
Gaussians evaluated on the source-plane grid.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
__Lens Light__

Compute a 2D image of each lens galaxy's light and sum them together.

For an MGE, each galaxy contributes many Gaussian images (e.g. 20 for the main lens, 10 per extra galaxy).
The total lens light image is the sum of ALL individual Gaussian images from ALL lens galaxies. With linear
light profiles, the intensity of each Gaussian is optimized to minimize the chi-squared.
"""
lens_image_2d = lens_galaxy.image_2d_from(grid=masked_dataset.grid)
extra_0_image_2d = extra_galaxy_0.image_2d_from(grid=masked_dataset.grid)
extra_1_image_2d = extra_galaxy_1.image_2d_from(grid=masked_dataset.grid)

total_lens_image_2d = lens_image_2d + extra_0_image_2d + extra_1_image_2d

aplt.plot_array(array=total_lens_image_2d, title="Total Lens Light (All Galaxies)")

"""
Compute blurring images for ALL lens galaxies.
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
__Lens Galaxy Mass__

We compute the deflection angles from all mass profiles in the group.
"""
deflections_lens = lens_galaxy.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_0 = extra_galaxy_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_1 = extra_galaxy_1.deflections_yx_2d_from(grid=masked_dataset.grid)

"""
__Ray Tracing__

The total deflection angle is the sum of deflection angles from ALL galaxies in the group.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

traced_blurring_grid = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.blurring
)[-1]

"""
__Source Image__

Evaluate the source galaxy light on the traced grid.

For an MGE source, the source light would be the sum of ~20 Gaussian images evaluated on the traced grid,
with intensities solved via the same linear algebra inversion.
"""
source_image_2d = source_galaxy.image_2d_from(grid=traced_grid)

source_blurring_image_2d = source_galaxy.image_2d_from(grid=traced_blurring_grid)

"""
__Lens + Source Light Addition__

Add the total lens light and source image together.
"""
image = total_lens_image_2d + source_image_2d

aplt.plot_array(array=image, title="Total Image (All Galaxies + Source)")

blurring_image_2d = total_lens_blurring_image_2d + source_blurring_image_2d

"""
__Convolution__

Convolve the 2D image with the PSF in real-space.
"""
convolved_image_2d = masked_dataset.psf.convolved_image_from(
    image=image, blurring_image=blurring_image_2d
)

aplt.plot_array(array=convolved_image_2d, title="Convolved Image")

"""
__Likelihood Function__

We now quantify the goodness-of-fit of our group-scale lens model.

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

__Chi Squared__
"""
model_image = convolved_image_2d

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

chi_squared_map = al.Array2D(values=chi_squared_map, mask=mask)

aplt.plot_array(array=chi_squared_map, title="Chi-Squared Map")

"""
__Noise Normalization Term__
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(figure_of_merit)

"""
__MGE Likelihood__

When using MGE light profiles (linear light profiles), the likelihood calculation differs in an important way.
Instead of each galaxy having a single light profile with a fixed intensity, each galaxy contributes a set of
Gaussian basis functions to a `mapping_matrix`.

The mapping matrix has dimensions `(total_image_pixels, total_linear_light_profiles)`, where each column
is the PSF-convolved image of one Gaussian. For a group with a 20-Gaussian main lens, two 10-Gaussian extras,
and a 20-Gaussian source, this matrix has 60 columns.

The intensities of all 60 Gaussians are then solved simultaneously via positive-only linear algebra:

  s = F^{-1} D

where F is the curvature matrix and D is the data vector (see the `linear_light_profiles` feature for details).

This joint optimization across all galaxies is what makes MGE so powerful for groups: the linear algebra
automatically determines the optimal intensity decomposition for every galaxy simultaneously, accounting for
blending between overlapping galaxies.

__Fit__

The `FitImaging` object handles all of the steps above automatically.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

We have presented a visual step-by-step guide to the group-scale parametric likelihood function, with
explanations of how MGE light profiles modify each step. The key differences when using MGE are:

 - Each galaxy's light is a sum of many Gaussians rather than a single analytic profile.
 - The intensities of all Gaussians across all galaxies are solved jointly via linear algebra.
 - This joint linear inversion adds zero non-linear parameters per galaxy, making it ideal for groups
   with many extra galaxies.
"""
