"""
Source Science: Pixelization (Group)
====================================

Source science focuses on studying the highly magnified properties of the background lensed source galaxy.

Using a pixelized source reconstruction from a group-scale lens, we can compute key quantities such as:

 - The total flux of the reconstructed source.
 - The magnification of the source due to the combined mass of all group galaxies.
 - The intrinsic size and morphology of the source.

For group-scale lenses, ALL mass profiles in the group (main lens + extra galaxies) contribute to the
magnification. Omitting any galaxy's mass would give an incorrect magnification estimate.

This script also compares the pixelized source flux to a parametric estimate, demonstrating that the
pixelized reconstruction can recover source properties that parametric models may miss.

__Contents__

**Dataset & Mask:** Standard set up of the group dataset and 7.5" mask.
**Galaxy Centres:** Load centres for main lens and extra galaxies.
**Model Fit:** Create a FitImaging with a pixelized source.
**Source Flux:** Compute the total flux from the pixelized source reconstruction.
**Source Magnification:** Compute the magnification using all group mass profiles.
**Impact of Extra Galaxies:** Demonstrate how omitting extra galaxies affects magnification.
**Interpolated Source:** Interpolate the pixelized source to a uniform grid.
**Parametric Comparison:** Compare pixelized source flux to a parametric estimate.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# from autoconf import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load the strong lens group dataset `simple`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__
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
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Galaxy Centres__
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")
extra_galaxies_centres = al.from_json(
    file_path=dataset_path / "extra_galaxies_centres.json"
)

"""
__Over Sampling__
"""
all_centres = list(main_lens_centres) + list(extra_galaxies_centres)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=all_centres,
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

"""
__Model Fit__

We create a fit with a pixelized source using concrete galaxy objects.

For the group lens, we include the main lens galaxy, extra galaxies, and a pixelized source.
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

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Constant(coefficient=1.0)

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(
    galaxies=[lens_galaxy, extra_galaxy_0, extra_galaxy_1, source_galaxy]
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
__Inversion__

The inversion contains all information about the pixelized source reconstruction.
"""
inversion = fit.inversion

mapper = inversion.cls_list_from(cls=al.Mapper)[0]

"""
__Source Flux__

The total flux of the pixelized source reconstruction is the sum of all source pixel fluxes.

The units are the same as the data (typically electrons per second, e- s^-1).
"""
reconstruction = inversion.reconstruction

total_source_flux = np.sum(reconstruction)

print(f"Total Source Flux via Pixelization: {total_source_flux} e- s^-1")

"""
The source pixel positions in the source plane are also available.
"""
source_plane_mesh_grid = mapper.source_plane_mesh_grid

print(f"Source Plane Mesh Grid: {source_plane_mesh_grid}")

"""
__Source Magnification__

The overall magnification of the source is the ratio of total flux in the image plane to total flux
in the source plane.

The image-plane reconstruction (mapped back from source pixels) is available from the inversion.
"""
mapped_reconstructed_operated_data = inversion.mapped_reconstructed_operated_data

"""
To compute the magnification, we need the areas of source and image pixels.

The image-plane pixels have a uniform area defined by the dataset pixel scale.
"""
from scipy.interpolate import griddata

interpolation_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

interpolated_reconstruction = griddata(
    points=source_plane_mesh_grid, values=reconstruction, xi=interpolation_grid
)

interpolated_reconstruction_ndarray = interpolated_reconstruction.reshape(
    interpolation_grid.shape_native
)

interpolated_reconstruction = al.Array2D.no_mask(
    values=interpolated_reconstruction_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

magnification = np.sum(
    mapped_reconstructed_operated_data * mapped_reconstructed_operated_data.pixel_area
) / np.sum(interpolated_reconstruction * interpolated_reconstruction.pixel_area)

print(f"Source Magnification (all group galaxies): {magnification}")

"""
__Impact of Extra Galaxies__

For group-scale lenses, ALL mass profiles contribute to the magnification. Omitting extra galaxies
gives an incorrect estimate.

Below, we compute the magnification using only the main lens galaxy for comparison.
"""
tracer_main_only = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit_main_only = al.FitImaging(dataset=dataset, tracer=tracer_main_only)

inversion_main_only = fit_main_only.inversion
reconstruction_main_only = inversion_main_only.reconstruction
mapped_main_only = inversion_main_only.mapped_reconstructed_operated_data

mapper_main_only = inversion_main_only.cls_list_from(cls=al.Mapper)[0]
source_plane_mesh_grid_main_only = mapper_main_only.source_plane_mesh_grid

interpolated_main_only = griddata(
    points=source_plane_mesh_grid_main_only,
    values=reconstruction_main_only,
    xi=interpolation_grid,
)

interpolated_main_only_ndarray = interpolated_main_only.reshape(
    interpolation_grid.shape_native
)

interpolated_main_only = al.Array2D.no_mask(
    values=interpolated_main_only_ndarray,
    pixel_scales=interpolation_grid.pixel_scales,
)

magnification_main_only = np.sum(
    mapped_main_only * mapped_main_only.pixel_area
) / np.sum(interpolated_main_only * interpolated_main_only.pixel_area)

print(f"Source Magnification (main lens only): {magnification_main_only}")
print(
    f"Magnification difference when omitting extra galaxies: "
    f"{magnification - magnification_main_only:.4f}"
)

"""
__Interpolated Source__

For detailed source science, the pixelized source can be interpolated to a uniform 2D grid.
This allows standard image analysis tools to be used.
"""
aplt.plot_array(array=interpolated_reconstruction, title="Interpolated Source")

"""
__Zoom__

We can zoom in on the source region for higher resolution.
"""
extent = (-1.0, 1.0, -1.0, 1.0)
shape_native = (401, 401)

interpolation_grid_zoom = al.Grid2D.from_extent(
    extent=extent,
    shape_native=shape_native,
)

interpolated_reconstruction_zoom = griddata(
    points=source_plane_mesh_grid,
    values=reconstruction,
    xi=interpolation_grid_zoom,
)

interpolated_reconstruction_zoom_ndarray = interpolated_reconstruction_zoom.reshape(
    interpolation_grid_zoom.shape_native
)

interpolated_reconstruction_zoom = al.Array2D.no_mask(
    values=interpolated_reconstruction_zoom_ndarray,
    pixel_scales=interpolation_grid_zoom.pixel_scales,
)

aplt.plot_array(
    array=interpolated_reconstruction_zoom, title="Zoomed Interpolated Source"
)

"""
__Errors__

The reconstruction noise map provides errors on each source pixel, enabling uncertainty propagation
for source science calculations.
"""
reconstruction_noise_map = inversion.reconstruction_noise_map

interpolated_noise_map = griddata(
    points=source_plane_mesh_grid,
    values=reconstruction_noise_map,
    xi=interpolation_grid,
)

interpolated_noise_map_ndarray = interpolated_noise_map.reshape(
    interpolation_grid.shape_native
)

interpolated_noise_map = al.Array2D.no_mask(
    values=interpolated_noise_map_ndarray, pixel_scales=interpolation_grid.pixel_scales
)

aplt.plot_array(array=interpolated_noise_map, title="Source Reconstruction Noise Map")

"""
__Parametric Comparison__

For comparison, we compute the source flux using a parametric source model (Sersic) that
approximates the true source used to simulate the dataset.

The pixelized reconstruction may recover more flux than the parametric model if the source has
complex morphology that a Sersic profile cannot capture.
"""
source_parametric = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.02)

parametric_image = source_parametric.bulge.image_2d_from(grid=grid)
total_parametric_flux = np.sum(parametric_image)

print(f"Total Source Flux (Parametric Sersic): {total_parametric_flux} e- s^-1")
print(f"Total Source Flux (Pixelized): {total_source_flux} e- s^-1")
print(
    f"Flux Difference (Pixelized - Parametric): "
    f"{total_source_flux - total_parametric_flux:.4f} e- s^-1"
)

"""
__Magnification via Mesh__

For more accurate magnification, we can use the mesh pixel areas directly rather than interpolating.
"""
mesh_areas = mapper.mesh_geometry.areas_for_magnification

magnification_mesh = np.sum(
    mapped_reconstructed_operated_data * mapped_reconstructed_operated_data.pixel_area
) / np.sum(reconstruction * mesh_areas)

print(f"Magnification via Mesh Areas: {magnification_mesh}")

"""
__Wrap Up__

This script demonstrated source science calculations using a pixelized source reconstruction for a
group-scale lens.

Key points:
 - The total source flux is the sum of all reconstructed source pixel values.
 - Magnification is computed as the ratio of image-plane to source-plane flux.
 - ALL group mass profiles must be included for accurate magnification.
 - The interpolated source can be analyzed using standard image tools.
 - Pixelized reconstructions may recover more flux than parametric models for complex sources.
"""
