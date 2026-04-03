"""
Plots: Pixelization
===================

This example illustrates the plotting API for pixelized source reconstructions.

The new API uses:

 - `aplt.plot_array()` — plot any 2D array (including source reconstructions).
 - `aplt.plot_grid()` — plot a grid of coordinates.
 - `aplt.subplot_fit_imaging()` — multi-panel fit overview.
 - `aplt.subplot_fit_interferometer()` — interferometer fit overview.

Inversion and mapper quantities are accessed via `fit.inversion` and plotted with `aplt.plot_array()`.

__Start Here Notebook__

Refer to `plots/start_here.ipynb` for an introduction to the new plotting API.

__Contents__

- **Setup**: Set up dataset, tracer and fit with a pixelized source.
- **Fit Imaging**: Plot the fit and its pixelized source reconstruction.
- **Inversion**: Plot the inversion reconstruction directly.
- **Mapper Grids**: Plot the image-plane and source-plane mesh grids.
- **Fit Interferometer**: Plot an interferometer fit with a pixelized source.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Setup__

Set up the dataset and a fit with a pixelized source reconstruction.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not dataset_path.exists():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/howtolens/simulator/lens_sersic.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Imaging__

Plot the multi-panel fit overview with `aplt.subplot_fit_imaging()`.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
Plot individual fit attributes.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
The pixelized source reconstruction is accessed via `fit.model_images_of_planes_list[1]`,
which is the reconstructed image of the source plane.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Source Plane Reconstruction",
)

"""
__Inversion__

The `inversion` property contains the linear algebra, mesh calculations and other key quantities
used to reconstruct the source galaxy.

The reconstruction is accessed via `fit.inversion.reconstruction`.
"""
inversion = fit.inversion

aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Inversion Reconstruction",
)

"""
An inversion can also be computed directly from a `Tracer` using `TracerToInversion`.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Inversion Reconstruction (via TracerToInversion)",
)

"""
__Mapper Grids__

The mapper maps pixels from the image-plane to the source-plane pixelization.

We can extract the image-plane and source-plane mesh grids and plot them as overlays.
"""
mapper = inversion.cls_list_from(cls=al.Mapper)[0]

image_plane_mesh_grid = mapper.mask.derive_grid.unmasked

aplt.plot_array(
    array=fit.data,
    title="Data with Image-Plane Mesh Grid",
    positions=image_plane_mesh_grid,
)

source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]

aplt.plot_grid(
    grid=source_plane_mesh_grid,
    title="Source-Plane Mesh Grid",
)

"""
__Mapper Galaxy Dict__

The mapper galaxy dict maps each mapper to its corresponding galaxy.
"""
mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict

mapper = list(mapper_galaxy_dict)[0]

"""
Plot the image-plane mesh grid and source-plane mesh grid together.
"""
image_plane_mesh_grid = mapper.mask.derive_grid.unmasked
source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]

aplt.plot_array(
    array=fit.data,
    title="Data with Mesh Grid Overlay",
    positions=image_plane_mesh_grid,
)

aplt.plot_grid(
    grid=source_plane_mesh_grid,
    title="Source-Plane Mesh Grid",
)

"""
__Fit Interferometer__

A fit to an interferometer dataset with a pixelized source is plotted with
`aplt.subplot_fit_interferometer()`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

aplt.subplot_fit_interferometer(fit=fit)

"""
Plot the dirty model image (the model image in real space for an interferometer fit).
"""
aplt.plot_array(
    array=fit.dirty_model_image,
    title="Dirty Model Image (Interferometer)",
)

"""
Finish.
"""
