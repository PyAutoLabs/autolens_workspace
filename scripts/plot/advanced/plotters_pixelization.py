"""
Plots: Plotters Pixelization
============================

This example illustrates the API for plotting using `Plotter` objects for pixelized source reconstructions.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, tracer, data) used to illustrate plotting.
**Fit Imaging:** Plot the fit of a tracer to an imaging dataset for a source reconstruction using a pixelization.
**Inversion:** Plot the inversion object which performs the linear algebra and other calculations which reconstruct the source galaxy.
**Mapper:** Plot the mapper object which maps pixels from the image-plane of the data to its source plane pixelization via a lens model.
**Fit Interferometer:** Plot the fit of a tracer to an interferometer dataset for a source reconstruction using a pixelization.

__Setup__

To illustrate plotting, we require standard objects like a grid, tracer and dataset.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
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
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Fit Imaging__

The `FitImaging` object is a base object which represents the fit of a model to an imaging dataset, including the
residuals, chi-squared and model image.

We plot the source-plane, which being pixelized, is represented by a `Pixelization` object and plotted as a 
delunay mesh of triangles.

The plot below zooms into the brightest pixel of the source-plane, which is useful for visualizing the key regions
of the source that fit the data.
"""
fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
)
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=True
)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`. 

This shows the full extent of the source-plane pixelization and may also include the caustics which the zoomed 
image does not due to zooming inside of them. This can be useful for ensuring that the construction of the
source-plane pixelization is reasonable.
"""
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=False
)

"""
An irregular mesh like the Delaunay or Voronoi can be plotted in two ways, using the irregular grid of cells or
by interpolating the reconstructed source-plane image onto a uniform grid of pixels.

By default, the irregular grid is plotted, but the interpolated image can be plotted by changing the
`interpolate_to_uniform` input to `True`.
"""
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, interpolate_to_uniform=True
)

"""
The mappings subplot shows the mappings between the image and source plane, by drawing circles around the brightest
source pixels and showing how they map to the image-plane.
"""
fit_plotter.subplot_mappings_of_plane(plane_index=1)

"""
The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
mapper = fit.inversion.cls_list_from(cls=al.AbstractMapper)[0]

image_plane_mesh_grid = mapper.image_plane_mesh_grid
visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)
fit_plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals_2d)
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)

source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]
visuals_2d = aplt.Visuals2D(mesh_grid=source_plane_mesh_grid)
fit_plotter = aplt.FitImagingPlotter(fit=fit, visuals_2d=visuals_2d)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
We can extract an `InversionPlotter` (described below) from the `FitImagingPlotter` and use it to plot all of its usual 
methods, which will now include the critical curves, caustics and border.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
__Inversion__

The fit above has a property called an `inversion`, which contains all of the linear algebra, mesh calculations
and other key quantities used to reconstruct a source galaxy using a pixelization.

This has its own dedicated plotter, the `InversionPlotter`, which can be used to plot the inversion's attributes
and properties in a similar way to the `FitImagingPlotter`.
"""
inversion = fit.inversion

inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_image=True)

"""
An inversion can also be computed directly from a `Tracer` object, using the `TracerToInversion` class.

The `FitImaging` object uses a `TracerToInversion` internally so the `inversion` objects are identical, but as a user
knowing both APIs could be useful.
"""
tracer_to_inversion = al.TracerToInversion(
    tracer=tracer,
    dataset=dataset,
)

inversion = tracer_to_inversion.inversion

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
    reconstruction_noise_map=True,
    regularization_weights=True,
)

"""
The `Inversion` attributes can also be plotted as a subplot.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.subplot_of_mapper(mapper_index=0)

"""
The mappings subplot shows the mappings between the image and source plane, by drawing circles around the brightest
source pixels and showing how they map to the image-plane.
"""
inversion_plotter.subplot_mappings(pixelization_index=0)

"""
The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

image_plane_mesh_grid = mapper.image_plane_mesh_grid
visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)
inversion_plotter = aplt.InversionPlotter(inversion=inversion, visuals_2d=visuals_2d)
inversion_plotter.figures_2d(reconstructed_image=True)

source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]
visuals_2d = aplt.Visuals2D(mesh_grid=source_plane_mesh_grid)
inversion_plotter = aplt.InversionPlotter(inversion=inversion, visuals_2d=visuals_2d)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Mapper__

The `Mapper` is a property of an inversion and maps pixels from the image-plane of the data to its source plane via 
a lens model.

We can extract a dictionary where every mapper in the plane is a key, paired with values that are each corresponding 
galaxy containing that mapper. 
"""
mapper_galaxy_dict = tracer_to_inversion.mapper_galaxy_dict


"""
We only need the `Mapper`, which we can extract from this dictionary.
"""
mapper = list(mapper_galaxy_dict)[0]

"""
We now pass the mapper to a `MapperPlotter` and call various `figure_*` methods to plot different attributes.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.figure_2d()

"""
The `Mapper` can also be plotted with a subplot of its original image.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
The Indexes of `Mapper` plots can be highlighted to show how certain image pixels map to the source plane.
"""
visuals = aplt.Visuals2D(indexes=[0, 1, 2, 3, 4])

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
The index of source plane pixels can be mapped to the image-plane to show mappings of source to image pixels.

The pixels, plotted in red, extended beyond the central square pixel of the source-plane grid. This is because
the pairing of data pixels to source pixels is not one-to-one, as an interpolation scheme is used to map pixels
which land near the edges of the source-pixel, but outside them, to that source pixel with a weight.
"""
pix_indexes = [[312, 318], [412]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
image_plane_mesh_grid = mapper.image_plane_mesh_grid
source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]

visuals_2d = aplt.Visuals2D(
    grid=image_plane_mesh_grid, mesh_grid=source_plane_mesh_grid
)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals_2d)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Fit Interferometer__

The `FitInterferometer` object is a base object which represents the fit of a model to an interferometer dataset,
including the residuals, chi-squared and model image.

We now create one which uses a pixelized source reconstruction.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "interferometer" / dataset_name

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=Path(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
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
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitInterferometer(dataset=dataset, tracer=tracer)

"""
The visualization plottes the reconstructed source on the Delaunay mesh, and you'll have seen it zoomed in to
its brightest pixels. 

This is so the galaxy can be clearly seen and is the default behavior of the `InversionPlotter`, given the
input `zoom_to_brightest=True`.
"""
fit_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
)
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=True
)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`.

This shows the full extent of the source-plane pixelization and may also include the caustics which the zoomed
image does not due to zooming inside of them. This can be useful for ensuring that the construction of the
source-plane pixelization is reasonable.
"""
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=False
)

"""
The mappings subplot shows the mappings between the image and source plane, by drawing circles around the brightest
source pixels and showing how they map to the image-plane.
"""
fit_plotter.subplot_mappings_of_plane(plane_index=1)

"""
The image and source plane mesh grids, showing the centre of every source pixel in the image-plane and source-plane, 
can be computed and plotted.
"""
mapper = fit.inversion.cls_list_from(cls=al.AbstractMapper)[0]

image_plane_mesh_grid = mapper.image_plane_mesh_grid
visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, visuals_2d=visuals_2d)
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)

source_plane_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_plane_mesh_grid)[-1]
visuals_2d = aplt.Visuals2D(mesh_grid=source_plane_mesh_grid)
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, visuals_2d=visuals_2d)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)


"""
We can even extract an `InversionPlotter` from the `FitInterferometerPlotter` and use it to plot all of its usual 
methods,which will now include the critical curves, caustics and border.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
__DelaunayDrawer / VoronoiDrawer__

We can customize the filling of Voronoi cells using the `VoronoiDrawer` object which wraps the 
method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
delaunay_drawer = aplt.DelaunayDrawer(edgecolor="b", linewidth=1.0, linestyle="--")
# voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b", linewidth=1.0, linestyle="--")

mat_plot = aplt.MatPlot2D(delaunay_drawer=delaunay_drawer)

"""
We now pass the inversion to a `InversionPlotter` which we will use to illustrate customization with 
the `VoronoiDrawer` object.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot)

try:
    inversion_plotter.figures_2d_of_pixelization(
        pixelization_index=0, reconstruction=True
    )
    inversion_plotter.subplot_of_mapper(mapper_index=0)
except ImportError:
    print(
        "You have not installed the Voronoi natural neighbor interpolation package, see instructions at top of notebook."
    )

"""
Finish.
"""
