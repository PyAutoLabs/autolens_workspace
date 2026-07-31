"""
Plots: Pixelization (Multi Galaxy)
==================================

This script plots a multi-galaxy fit whose source is reconstructed on a pixelized mesh.

The plotting functions are the same ones used everywhere else in the workspace (see
`multi_galaxy/plot.py`):

 - `aplt.plot_array()` — plot any 2D array, including source-plane model images.
 - `aplt.plot_grid()` — plot a grid of coordinates, including source-plane mesh grids.
 - `aplt.subplot_fit_imaging()` — the multi-panel fit overview.

What a pixelization adds is the `inversion` on the fit, and the `Mapper` inside it, which is what the mesh-grid
plots below come from.

__Contents__

- **Setup:** Load the dataset and build a fit with a pixelized source.
- **Fit Imaging:** The multi-panel overview and the individual fit attributes.
- **Inversion:** The reconstruction, and building an inversion without a fit.
- **Mapper Grids:** The image-plane and source-plane mesh grids.
- **Mapper Galaxy Dict:** Pairing each mapper to its galaxy.
- **Wrap Up:** Where to go next.

__Start Here Notebook__

For an introduction to the plotting API refer to `guides/plot/start_here.py`. For the multi-galaxy plotting
conventions — including the plane images of a lens with two deflectors — refer to `multi_galaxy/plot.py`.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Setup__

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

"""
The faint contaminant's light is scaled out of the fit, as `multi_galaxy/modeling.py` explains.
"""
mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=dataset.pixel_scales,
    invert=True,
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask_radius = 3.0

dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
)

"""
The two co-dominant deflectors, and a source with a pixelization instead of a light profile.
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

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, shear_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Imaging__

The multi-panel fit overview.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
The individual fit attributes, each a 2D array.
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.model_data, title="Model Image")
aplt.plot_array(array=fit.residual_map, title="Residual Map")
aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")
aplt.plot_array(array=fit.chi_squared_map, title="Chi-Squared Map")

"""
`model_images_of_planes_list` holds one model image per plane. Both deflectors are at the same redshift, so
there are two planes here regardless of how many deflectors the lens has: index 0 is the lens plane, holding
both deflectors' light summed, and index 1 is the source plane, holding the reconstruction mapped back to the
image plane.
"""
aplt.plot_array(
    array=fit.model_images_of_planes_list[0],
    title="Lens Plane (Both Deflectors)",
)

aplt.plot_array(
    array=fit.model_images_of_planes_list[1],
    title="Source Plane Reconstruction",
)

"""
__Inversion__

The `inversion` holds the linear algebra and mesh calculations behind the reconstruction.

`inversion.reconstruction` is the raw flux of every source pixel. It lives on the source-plane mesh rather than
on a uniform 2D grid, which is why the figure above plots the source-plane model image instead of it directly.
`multi_galaxy/features/pixelization/source_science.py` shows how to interpolate it onto a regular grid.
"""
inversion = fit.inversion

print(inversion.reconstruction)

"""
An inversion can also be built straight from a `Tracer` with `TracerToInversion`, without a fit — useful for
inspecting a pixelization before fitting anything.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
__Mapper Grids__

The `Mapper` maps image-plane pixels to source-plane pixels. Its image-plane mesh grid can be overlaid on the
data, and ray-traced to the source plane to show where those coordinates land.

The trace uses the summed deflection of both deflectors, so the source-plane grid below reflects both mass
models rather than either one.
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

A model with more than one pixelized galaxy has more than one mapper. `mapper_galaxy_dict` pairs them, keyed by
the mapper, and the plots above can be repeated for any entry in it.

This lens has one pixelized galaxy — the source — so there is one mapper. The deflectors carry light profiles,
not pixelizations, and so do not appear here.
"""
mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict

mapper = list(mapper_galaxy_dict)[0]

print(mapper_galaxy_dict[mapper])

"""
__Wrap Up__

Where to go next:

 - `multi_galaxy/plot.py` — the plotting conventions for a multi-galaxy fit, without a pixelization.
 - `multi_galaxy/features/pixelization/fit.py` — the fit the objects plotted here come from.
 - `multi_galaxy/features/pixelization/source_science.py` — measuring the reconstruction rather than plotting it.
 - `imaging/features/pixelization/plot.py` — the galaxy-scale version, which also covers interferometer fits.
"""
