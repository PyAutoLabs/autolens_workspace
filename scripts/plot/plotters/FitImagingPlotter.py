"""
Plots: FitImagingPlotter
========================

This example illustrates how to plot an `FitImaging` object using an `FitImagingPlotter`.

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
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Fit__

We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
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

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Figures__

We now pass the FitImaging to an `FitImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)

fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
It can plot of the model image of an input plane.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, model_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, model_image=True)

"""
It can plot the image of a plane with all other model images subtracted.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, subtracted_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)

"""
__Source Zoom__

It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image).

By default, this source-plane image is zoomed to the brightest pixels, so the galaxy can be clearly seen.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=True
)

"""
If we do not want the image to be zoomed, we can pass `zoom_to_brightest=False`.

This shows the full extent of the grid used to create the source-plane image, and may also include the caustics
which the zoomed image does not due to zooming inside of them.
"""
fit_plotter.figures_2d_of_planes(
    plane_index=1, plane_image=True, zoom_to_brightest=False
)

"""
__Source Brightness__

The source is often much fainter than the lens galaxy, meaning we may want to brighten its appearance to fully see it.

We can do this by passing the `use_source_vmax` bool, which sets the maximum value of the colormap to the maximum
flux in the source-plane.

This is used by default in the subplots plotted below.
"""
fit_plotter.figures_2d(data=True, use_source_vmax=True)
fit_plotter.figures_2d_of_planes(plane_index=1, model_image=True, use_source_vmax=True)
fit_plotter.figures_2d_of_planes(
    plane_index=1, subtracted_image=True, use_source_vmax=True
)

"""
__Subplots__

The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
It also includes a log10 subplot option, which shows the same figures but with the colormap in log10 format to
highlight the fainter regions of the data.
"""
fit_plotter.subplot_fit_log10()

"""`
__Include__

`FitImaging` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
    tangential_caustics=True,
    radial_caustics=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=0)
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Symmetric Residual Maps__

By default, the `residual_map` and `normalized_residual_map` use a symmetric colormap.

This means the maximum normalization (`vmax`) an minimum normalziation (`vmin`) are the same absolute value.

This can be disabled via the `residuals_symmetric_cmap` input.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, residuals_symmetric_cmap=False)
fit_plotter.figures_2d(
    residual_map=True,
    normalized_residual_map=True,
)

"""
__Pixelization__

We can also plot a `FitImaging` which uses a `Pixelization`.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Reconstruction Options__

The visualization plottes the reconstructed source on the Delaunay mesh, and you'll have seen it zoomed in to
its brightest pixels. 

This is so the galaxy can be clearly seen and is the default behavior of the `InversionPlotter`, given the
input `zoom_to_brightest=True`.
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
__Include__

It can use the `Include2D` object to plot the `Mapper`'s specific structures like the image and source plane 
pixelization grids.
"""
include = aplt.Include2D(
    mapper_image_plane_mesh_grid=True, mapper_source_plane_data_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
In fact, via the `FitImagingPlotter` we can plot the `reconstruction` with caustics and a border, which are extracted
from the `Tracer` of the `FitImaging`.

To do this with an `InversionPlotter` we would have had to manually pass these attributes via the `Visuals2D` object.
"""
include = aplt.Include2D(
    border=True,
    tangential_caustics=True,
    radial_caustics=True,
    mapper_image_plane_mesh_grid=True,
    mapper_source_plane_data_grid=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
__Inversion Plotter__

We can even extract an `InversionPlotter` from the `FitImagingPlotter` and use it to plot all of its usual methods,
which will now include the critical curves, caustics and border.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
__Double Einstein Ring__

The `FitImagingPlotter` can also plot lens systems with two (or more) Einstein rings.

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
__Fit__

We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
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
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.15, -0.15), intensity=1.2, effective_radius=0.1
    ),
    mass=al.mp.IsothermalSph(centre=(-0.15, -0.15), einstein_radius=0.3),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(-0.45, 0.45), intensity=0.6, effective_radius=0.07
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)


"""
__Figures__

We now pass the FitImaging to an `FitImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit)

fit_plotter.set_mat_plots_for_subplot(is_for_subplot=False)

fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
It can plot of the model image of an input plane.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, model_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, model_image=True)
fit_plotter.figures_2d_of_planes(model_image=True, plane_index=2)

"""
It can plot the image of a plane with all other model images subtracted.
"""
fit_plotter.figures_2d_of_planes(plane_index=0, subtracted_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, subtracted_image=True)
fit_plotter.figures_2d_of_planes(subtracted_image=True, plane_index=2)

"""
It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image)
"""
fit_plotter.figures_2d_of_planes(plane_index=0)
fit_plotter.figures_2d_of_planes(plane_index=1)
fit_plotter.figures_2d_of_planes(plane_index=2)

"""
__Subplots__

The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=0)
fit_plotter.subplot_of_planes(plane_index=1)
fit_plotter.subplot_of_planes(plane_index=2)

"""`
__Include__

`FitImaging` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    tangential_critical_curves=True,
    radial_critical_curves=True,
    tangential_caustics=True,
    radial_caustics=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=0)
fit_plotter.subplot_of_planes(plane_index=1)
fit_plotter.subplot_of_planes(plane_index=2)

"""
__Pixelization__

We can also plot a `FitImaging` which uses a `Pixelization`.
"""
source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    pixelization=al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=2.0,
    pixelization=al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Include__

The `plane_image_from_plane` method now plots the reconstructed source on the Delaunay pixel-grid. It can use the
`Include2D` object to plot the `Mapper`'s specific structures like the image and source plane pixelization grids.
"""
include = aplt.Include2D(
    mapper_image_plane_mesh_grid=True, mapper_source_plane_data_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)
fit_plotter.figures_2d_of_planes(plane_image=True, plane_index=2)

"""
In fact, via the `FitImagingPlotter` we can plot the `reconstruction` with caustics and a border, which are extracted 
from the `Tracer` of the `FitImaging`. 

To do this with an `InversionPlotter` we would have had to manually pass these attributes via the `Visuals2D` object.
"""
include = aplt.Include2D(
    border=True,
    tangential_caustics=True,
    radial_caustics=True,
    mapper_image_plane_mesh_grid=True,
    mapper_source_plane_data_grid=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=0, plane_image=True)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)
fit_plotter.figures_2d_of_planes(plane_image=True, plane_index=2)

"""
__Inversion Plotter__

We can even extract an `InversionPlotter` from the `FitImagingPlotter` and use it to plot all of its usual methods, 
which will now include the caustic and border.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=2)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)
