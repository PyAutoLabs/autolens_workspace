"""
Plots: Pixelization
===================

This example illustrates how to plot fits which reconstruct the source galaxy using a pixelization.

Pixelized source reconstructions are plotted with the same functions as other fits (see
`scripts/imaging/plot.py`):

 - `aplt.plot_array()` — plot any 2D array (including source-plane model images).
 - `aplt.plot_grid()` — plot a grid of coordinates (including source-plane mesh grids).
 - `aplt.subplot_fit_imaging()` / `aplt.subplot_fit_interferometer()` — multi-panel fit overviews.

Inversion and mapper quantities are accessed via `fit.inversion` and visualized by plotting the
fit's source-plane model image and the mapper's mesh grids.

For an introduction to the plotting API refer to `guides/plot/start_here.py`.

__Contents__

- **Setup:** Set up the dataset and a fit with a pixelized source reconstruction.
- **Fit Imaging:** Plot the multi-panel fit overview and individual fit attributes.
- **Inversion:** Access the inversion's linear algebra and plot the source-plane model image.
- **Mapper Grids:** Plot the mapper's image-plane and source-plane mesh grids.
- **Mapper Galaxy Dict:** Access each mapper via its corresponding galaxy.
- **Fit Interferometer:** Plot a fit to an interferometer dataset with a pixelized source.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Setup__

Set up the dataset and a fit with a pixelized source reconstruction.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
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
The reconstructed source, mapped back to the image plane, is accessed via
`fit.model_images_of_planes_list[1]` — the model image of the source plane.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Source Plane Reconstruction",
)

"""
__Inversion__

The `inversion` property contains the linear algebra, mesh calculations and other key quantities
used to reconstruct the source galaxy.

The raw reconstruction — the value of every source-pixel in the mesh — is stored at
`fit.inversion.reconstruction`. It is defined on the (irregular) source-plane mesh rather than a
uniform 2D grid, which is why the figures here plot the source-plane model image above instead.
"""
inversion = fit.inversion

print(inversion.reconstruction)

"""
An inversion can also be computed directly from a `Tracer` using `TracerToInversion`, without
building a fit first — useful for inspecting a pixelization before fitting.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
__Mapper Grids__

The mapper maps pixels from the image-plane to the source-plane pixelization.

We can extract the image-plane mesh grid and overlay it on the data, and ray-trace it to the
source plane to plot the source-plane mesh grid.
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

When a model contains multiple pixelized galaxies, each mapper is paired to its galaxy via the
`mapper_galaxy_dict`, whose keys are the mappers and values the galaxies. The mesh-grid plots
above can be repeated for any mapper extracted from this dictionary.
"""
mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict

mapper = list(mapper_galaxy_dict)[0]

print(mapper_galaxy_dict[mapper])

"""
__Fit Interferometer__

A fit to an interferometer dataset with a pixelized source is plotted with
`aplt.subplot_fit_interferometer()`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator.py"],
        check=True,
    )

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
__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness.
The ENV line declares the environment applied when this script runs in CI
(PyAutoHands docs/env_profile_redesign.md §10); this whole section is
stripped from generated notebooks and markdown.

Guides load committed full-resolution FITS; SMALL_DATASETS would mismatch
the pre-existing 100x100 data shape.

ENV: full_datasets
"""
