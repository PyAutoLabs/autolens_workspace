"""
Source Science: Multi Gaussian Expansion (Group)
================================================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy.

Using a source galaxy model, we can compute key quantities such as the magnification, total flux, and intrinsic
size of the source.

This example shows how to perform these calculations for a group-scale lens using an MGE source model. The key
differences from the standard group source science example are:

 - The source galaxy is modeled as a Multi Gaussian Expansion (MGE), where the intensities of ~20 Gaussians are
   solved via linear algebra when fitting the data.
 - MGE sources can provide more accurate flux and magnification estimates than single parametric profiles (e.g.
   Sersic) because they capture more complex source morphologies.
 - ALL mass profiles in the group must be included when computing ray-tracing and magnification.

__Contents__

**Simulated Dataset:** Load the group dataset.
**Mask:** Define the 2D mask.
**Source Values:** Set up the lens and source galaxies.
**MGE Source:** Create an MGE source and fit it to the data to obtain intensities.
**Source Flux:** Compute the total source flux.
**Source Magnification:** Compute the source magnification using all group galaxies.
**Impact of Extra Galaxies:** Show that omitting extra galaxies gives incorrect magnification.
**Tracer:** Using the tracer from lens modeling results.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Simulated Dataset__

We load and plot the `simple` group example dataset.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

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

We apply a 7.5 arcsecond circular mask.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Load the centres of the main lens galaxies and extra galaxies.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__

Apply adaptive over-sampling at all galaxy centres.
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Source Values__

For a group-scale lens, we must include ALL galaxies that contribute to the lensing potential when
performing source science calculations. This includes the main lens galaxy and all extra galaxies.

We first set up the lens and extra galaxies with their mass profiles (no light, since we only need
the mass for ray-tracing in this context). We also create an MGE source whose intensities will be
determined by fitting the data.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

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
__MGE Source__

We set up the source galaxy using an MGE made of 20 Gaussians whose `sigma` values span 0.01" to the
mask radius. The intensities are linear light profiles whose values are determined by fitting the data.

The MGE source model offers advantages for source science calculations:

 - It captures more complex source morphologies than a single Sersic profile.
 - The intensities are optimized to best reconstruct the lensed source, providing a more accurate
   estimate of the source's total flux.
 - It naturally handles sources with multiple components or asymmetric features.
"""
total_gaussians = 20

log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

bulge_gaussian_list = []

for i in range(total_gaussians):
    gaussian = al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

bulge = al.lp_basis.Basis(profile_list=bulge_gaussian_list)

source_galaxy = al.Galaxy(redshift=1.0, bulge=bulge)

"""
We create the tracer using ALL group galaxies and the MGE source, then fit it to the data to
solve for the Gaussian intensities.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
From the fit, extract the tracer with solved-for Gaussian intensities.
"""
tracer = fit.model_obj_linear_light_profiles_to_light_profiles

aplt.subplot_fit_imaging(fit=fit)

"""
__Source Flux__

We compute the total source flux by summing the source galaxy's image over a high-resolution grid.
"""
source_galaxy = tracer.galaxies[-1]

grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_flux = np.sum(image)

print(f"Total Source Flux (MGE): {total_flux} e- s^-1")

"""
__Source Magnification__

The overall magnification is the ratio of total flux in the image-plane to the source-plane.

For group-scale lenses, the ray-tracing is performed through ALL mass profiles in the group, which
the tracer handles automatically.
"""
grid = al.Grid2D.uniform(shape_native=(1000, 1000), pixel_scales=0.03)

image = source_galaxy.bulge.image_2d_from(grid=grid)

total_source_plane_flux = np.sum(image)

traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

source_plane_grid = traced_grid_list[-1]

lensed_source_image = source_galaxy.bulge.image_2d_from(grid=source_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Magnification (all group galaxies, MGE): {source_magnification}")

"""
__Impact of Extra Galaxies__

For group-scale lenses, the magnification is determined by ALL mass profiles in the group. Omitting
the extra galaxy masses gives an incorrect magnification estimate.
"""
tracer_main_only = al.Tracer(galaxies=[tracer.galaxies[0], source_galaxy])

traced_grid_list_main_only = tracer_main_only.traced_grid_2d_list_from(grid=grid)

source_plane_grid_main_only = traced_grid_list_main_only[-1]

lensed_source_image_main_only = source_galaxy.bulge.image_2d_from(
    grid=source_plane_grid_main_only
)

total_image_plane_flux_main_only = np.sum(lensed_source_image_main_only)

source_magnification_main_only = (
    total_image_plane_flux_main_only / total_source_plane_flux
)

print(f"Source Magnification (main lens only, MGE): {source_magnification_main_only}")
print(
    f"Magnification difference when omitting extra galaxies: "
    f"{source_magnification - source_magnification_main_only:.4f}"
)

"""
__Tracer__

Lens modeling returns a `max_log_likelihood_tracer`, which is the object you would use to compute
source science calculations for real datasets. The code below reproduces the calculations above
using the tracer's built-in plane functionality.
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=grid)

image_plane_grid = traced_grid_list[0]
source_plane_grid = traced_grid_list[-1]

lensed_source_image = tracer.planes[-1].image_2d_from(grid=source_plane_grid)
source_plane_image = tracer.planes[-1].image_2d_from(grid=image_plane_grid)

total_image_plane_flux = np.sum(lensed_source_image)
total_source_plane_flux = np.sum(source_plane_image)

source_magnification = total_image_plane_flux / total_source_plane_flux

print(f"Source Plane Total Flux via Tracer (MGE): {total_source_plane_flux} e- s^-1")
print(f"Source Magnification via Tracer (MGE): {source_magnification}")

"""
__Parametric Source Models__

If your lens modeling uses a parametric source model (e.g. Sersic or MGE), the only object you need to
perform source science calculations is the `max_log_likelihood_tracer` returned by lens modeling.

For group-scale lenses, ensure that the tracer includes all group member galaxies with their mass profiles,
as each contributes to the ray-tracing and therefore to the magnification of the source.

An MGE source may give different flux and magnification estimates compared to a single Sersic source. If
precise source science calculations are important for your analysis, we recommend comparing results from
different source models (e.g. MGE, Sersic, pixelized reconstruction) to estimate systematic uncertainties.
"""
