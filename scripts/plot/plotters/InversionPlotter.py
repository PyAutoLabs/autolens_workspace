"""
Plots: InversionPlotter
=======================

This example illustrates how to plot a `Inversion` using a `InversionPlotter`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
dataset = dataset.apply_mask(mask=mask)

"""
__Tracer__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a lens model.

Lets create a `Tracer` which we will use to create the `Inversion`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp_linear.Sersic(),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Converting a `Tracer` to an `Inversion` performs a number of steps, which are handled by the `TracerToInversion` class. 

This class is where the data and tracer's galaxies are combined to fit the data via the inversion.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
__Figures__

We now pass the inversion to a `InversionPlotter` and call various `figure_*` methods to plot different attributes.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An `Inversion` can have multiple mappers, which reconstruct multiple source galaxies at different redshifts and
planes (e.g. double Einstein ring systems).

To plot an individual source we must therefore specify the mapper index of the source we plot.
"""
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0,
    reconstructed_image=True,
    reconstruction=True,
    errors=True,
    regularization_weights=True,
)

"""
__Subplots__

The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
The mappings subplot shows the mappings between the image and source plane, by drawing circles around the brightest
source pixels and showing how they map to the image-plane.
"""
inversion_plotter.subplot_mappings(pixelization_index=0)

"""`
__Include__

Inversion`'s have their own unique attributes that can be plotted via the `Include2D` class:
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    mapper_image_plane_mesh_grid=True,
    mapper_source_plane_mesh_grid=True,
    mapper_source_plane_data_grid=True,
)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, include_2d=include)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)


"""
__Double Einstein Ring__

The `InversionPlotter` can also plot lens systems with two (or more) Einstein rings.

First, lets load example imaging of a strong lens with two Einstein rings.
"""
dataset_name = "double_einstein_ring"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the `Imaging` data so we can fit it with an `Inversion`.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.5,
)
dataset = dataset.apply_mask(mask=mask)

"""
__Tracer__

The `Inversion` maps pixels from the image-plane of our `Imaging` data to its source plane, via a lens model.

Lets create a `Tracer` which we will use to create the `Inversion`.
"""

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
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
    pixelization=pixelization,
)

source_galaxy_1 = al.Galaxy(redshift=2.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

"""
Converting a `Tracer` to an `Inversion` performs a number of steps, which are handled by the `TracerToInversion` class. 

This class is where the data and tracer's galaxies are combined to fit the data via the inversion.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

"""
__Figures__

We now pass the inversion to a `InversionPlotter` and call various `figure_*` methods to plot different attributes.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An `Inversion` can have multiple mappers, which reconstruct multiple source galaxies at different redshifts and
planes (e.g. double Einstein ring systems).

To plot an individual source we must therefore specify the mapper index of the source we plot.
"""
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstructed_image=True, reconstruction=True
)

inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=1, reconstructed_image=True, reconstruction=True
)

"""
__Subplots__

The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)
inversion_plotter.subplot_of_mapper(mapper_index=1)

"""`
__Include__

Inversion`'s have their own unique attributes that can be plotted via the `Include2D` class:
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    mapper_image_plane_mesh_grid=True,
    mapper_source_plane_mesh_grid=True,
    mapper_source_plane_data_grid=True,
)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, include_2d=include)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
Finish.
"""
