"""
__Log Likelihood Function: Linear Light Profiles (Group)__

This script provides a step-by-step guide of the ``log_likelihood_function`` which is used to fit ``Imaging``
data of a group-scale strong lens using **linear light profiles**.

The key difference from the standard group likelihood function is that the ``intensity`` of each light
profile is not a free parameter. Instead, intensities are solved via linear algebra (an "inversion") every
time the model is evaluated, always finding the intensity values that maximize the likelihood given all
other parameters.

This script has the following aims:

 - To provide a resource that authors can include in papers, so that readers can understand the likelihood
   function (including references to the previous literature from which it is defined) without having to
   write large quantities of text and equations.

 - To illustrate how the linear inversion works for group-scale lenses with multiple galaxies.

__Contents__

**Dataset & Mask:** Standard set up of the dataset and mask that is fitted.
**Over Sampling:** Set up the adaptive over-sampling grid for accurate light profile evaluation.
**Linear Light Profiles:** Define the lens, extra galaxies and source using ``lp_linear`` profiles.
**Lens Galaxy Mass:** Compute deflection angles from all mass profiles.
**Ray Tracing:** Trace image-plane coordinates to the source plane.
**Linear Inversion:** Explain how intensities are solved via the linear algebra system.
**Fit:** Use the ``FitImaging`` object which handles all steps automatically.
**Likelihood Function:** Compute the log likelihood step by step.

__Prerequisites__

The likelihood function of a linear light profile builds on that used for standard light profiles,
therefore you should read the following before this script:

- ``group/likelihood_function.py``
- ``imaging/features/linear_light_profiles/likelihood_function.py``
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

We define a 7.5" circular mask. This is larger than a typical galaxy-scale mask because group-scale systems
have lensed images that extend over a wider area.
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

For simplicity, we disable over sampling in this guide by setting ``sub_size=1``.
"""
masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Masked Image Grid__

The masked dataset provides a 2D grid of (y,x) coordinates used for all calculations.
"""
aplt.plot_grid(grid=masked_dataset.grids.lp, title="")

"""
__Linear Light Profiles__

We define all galaxies using linear light profiles from the ``lp_linear`` module. These profiles have no
``intensity`` parameter -- internally they use an intensity of 1.0, and the true intensity is solved via
linear algebra.

__Main Lens Galaxy__

The main lens galaxy has a spherical linear Sersic light profile and an isothermal mass profile.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(0.0, 0.0), effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

"""
__Extra Galaxies__

The two extra galaxies are companion galaxies near the main lens. They each have linear SersicSph light
profiles and isothermal mass profiles.
"""
extra_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(3.5, 2.5), effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(3.5, 2.5), einstein_radius=0.8),
)

extra_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.SersicSph(
        centre=(-4.4, -5.0), effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=(-4.4, -5.0), einstein_radius=1.0),
)

"""
__Source Galaxy Light Profile__

The source galaxy uses a linear cored elliptical Sersic -- again with no intensity parameter.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp_linear.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
Internally, linear light profiles have an ``intensity`` parameter set to 1.0. This is because each profile's
image (with unit intensity) forms a column in the "mapping matrix" used in the linear inversion.
"""
print(f"Lens bulge internal intensity: {lens_galaxy.bulge.intensity}")
print(f"Extra galaxy 0 internal intensity: {extra_galaxy_0.bulge.intensity}")
print(f"Source internal intensity: {source_galaxy.bulge.intensity}")

"""
__Lens Galaxy Mass__

We compute the deflection angles from ALL galaxies in the group. For group-scale lensing, the total
deflection is the sum of deflections from every galaxy's mass profile.
"""
deflections_lens = lens_galaxy.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_0 = extra_galaxy_0.deflections_yx_2d_from(grid=masked_dataset.grid)
deflections_extra_1 = extra_galaxy_1.deflections_yx_2d_from(grid=masked_dataset.grid)

"""
__Ray Tracing__

The total deflection is the sum of deflections from all galaxies. We ray-trace every image-plane
coordinate to the source plane using these combined deflections.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[-1]

aplt.plot_grid(grid=traced_grid, title="Source Plane Grid (Traced)")

"""
__Linear Inversion__

For standard light profiles, we would now evaluate each profile's image (with known intensity) and sum them
to get the model image. With linear light profiles, the process is different:

1. Each linear light profile's image is computed with unit intensity (intensity = 1.0). These images form
   the columns of a "mapping matrix" (or "blurred mapping matrix" after PSF convolution).

2. The mapping matrices from ALL planes (image-plane for lens galaxies, source-plane for the source) are
   combined into a single system.

3. A linear algebra solver finds the intensity values that minimize chi-squared. This is done using a
   positive-only solver to ensure all intensities are physical (non-negative).

4. The solved intensities are multiplied by the unit-intensity images to produce the final model image.

The ``FitImaging`` object handles all of this automatically. When it detects linear light profiles in the
tracer, it sets up and solves the linear system.
"""

"""
__Fit__

The ``FitImaging`` object performs the full likelihood evaluation, including the linear inversion. We can
verify that the figure of merit matches what we would compute manually.
"""
fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)
fit_figure_of_merit = fit.figure_of_merit
print(fit_figure_of_merit)

aplt.subplot_fit_imaging(fit=fit)

"""
__Likelihood Function__

The likelihood function for linear light profiles includes the same terms as the standard parametric case:

 $-2 \mathrm{ln} \, \epsilon = \chi^2 + \sum_{\rm  j=1}^{J} { \mathrm{ln}} \left [2 \pi (\sigma_j)^2 \right]  \, .$

The difference is that the model image used to compute the chi-squared is itself the output of the linear
inversion, rather than being computed from fixed intensity values.

__Chi Squared__

The chi-squared measures how well the model (with solved intensities) fits the data.
"""
model_image = fit.model_data
residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0
chi_squared = np.sum(chi_squared_map)

print(f"Chi-squared: {chi_squared}")

"""
__Noise Normalization Term__

The noise normalization term is the same as for standard light profiles.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

"""
__Calculate The Log Likelihood__

We compute the log likelihood by combining the chi-squared and noise normalization terms.
"""
figure_of_merit = float(-0.5 * (chi_squared + noise_normalization))

print(f"Log likelihood (manual): {figure_of_merit}")
print(f"Log likelihood (fit):    {fit_figure_of_merit}")

"""
__Lens Modeling__

To fit a lens model to data, the likelihood function illustrated in this tutorial is sampled using a
non-linear search algorithm.

The default sampler is the nested sampling algorithm ``Nautilus``. The linear inversion for intensities
happens automatically at every likelihood evaluation, meaning the non-linear search only needs to explore
the structural parameters (centres, effective radii, sersic indices, mass parameters, etc.).

__Wrap Up__

We have presented a visual step-by-step guide to the group-scale linear light profile likelihood function.

The key differences from the standard group likelihood function are:

 - Intensities are not free parameters but are solved via linear algebra at each likelihood evaluation.
 - The mapping matrix encodes each profile's unit-intensity image; the inversion finds the best intensities.
 - For group-scale lenses with many galaxies, this dramatically reduces the non-linear parameter space.
 - The ``FitImaging`` and ``Tracer`` objects handle all of this automatically.
"""
